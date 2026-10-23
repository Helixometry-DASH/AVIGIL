#!/usr/bin/env python
"""Cache VideoMAE V2 region features for one dataset split.

Writes one flat fp16 memmap per split plus a JSON index:

    <out>/<split>.f16          (total_positions, 1 + regions**2, 768)
    <out>/<split>_index.json   file -> (offset, length) and the build settings

A temporal position is one tubelet: ``TUBELET`` frames, 80 ms at 25 fps. A clip
of ``video_frames`` frames contributes ``video_frames // TUBELET`` positions, so
every offset is computable from the metadata alone.

Shards write disjoint byte ranges of the same file, so ``--rank``/``--world``
processes run concurrently on separate GPUs with no coordination.

    for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python scripts/extract_features.py \
        --root /path/to/dataset --split train --out /path/to/features \
        --rank $r --world 4 & done

Repeat for ``--split dev`` and ``--split test``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from avigil.models.backbone import (
    EMBED,
    TUBELET,
    WINDOW,
    center_weights,
    load_backbone,
    normalize,
    pool_regions,
    window_starts,
    window_tokens,
)


def allocate(ipath: Path, mpath: Path, index: dict, nbytes: int, rank: int) -> None:
    """Create the index and the sparse memmap exactly once, from any rank.

    Originally only rank 0 created these and the others slept up to 120 s for
    them to appear. That is a race whenever the ranks drift apart: on the 30%
    subset ranks 2 and 3 finished ``train`` about seven minutes before rank 0
    did, moved on to ``dev``, waited out the 120 s and died with
    ``FileNotFoundError: dev.f16``. The creator must not be a privileged rank.

    So every rank races to create, with ``O_CREAT | O_EXCL`` picking one winner,
    and the losers wait for the file to reach full size -- unbounded, because
    "the leader is still busy" is a normal state, not a failure. The index goes
    through a temp file and an atomic rename so no rank can read it half
    written.
    """
    if not ipath.exists():
        tmp = ipath.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(index, f)
        os.replace(tmp, ipath)          # atomic; last writer wins, same content

    try:
        fd = os.open(mpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as f:
            f.truncate(nbytes)          # sparse; shards fill disjoint ranges
        print(f"[rank{rank}] allocated {mpath} = {nbytes/2**30:.1f} GiB", flush=True)
        return

    waited = 0
    while not (mpath.exists() and mpath.stat().st_size == nbytes):
        time.sleep(2)
        waited += 2
        if waited % 60 == 0:
            print(f"[rank{rank}] waiting for {mpath.name} ({waited}s)", flush=True)


def build_index(records: list[dict], regions: int, stride: int) -> dict:
    offsets, total = {}, 0
    for r in sorted(records, key=lambda x: x["file"]):
        n = int(r["video_frames"]) // TUBELET
        offsets[r["file"]] = [total, n]
        total += n
    return {
        "offsets": offsets,
        "total_positions": total,
        "vectors_per_position": regions * regions + 1,
        "embed_dim": EMBED,
        "regions": regions,
        "tubelet": TUBELET,
        "window": WINDOW,
        "stride": stride,
        "fps": 25.0,
        "dtype": "float16",
        "layout": "channel 0 = global mean over 196 patches, 1: = regions**2 grid row-major",
    }


def decode_frames(path: str) -> np.ndarray:
    import av

    with av.open(path) as c:
        c.streams.video[0].thread_type = "AUTO"
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


class ClipFrames(torch.utils.data.Dataset):
    """Decodes one clip per item; decode is ~35 ms so workers keep the GPU fed."""

    def __init__(self, root: Path, files: list[str]):
        self.root, self.files = root, files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        f = self.files[i]
        return f, torch.from_numpy(decode_frames(str(self.root / f)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metadata", default="metadata.min.json")
    ap.add_argument("--regions", type=int, default=4)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(root / args.metadata) as f:
        records = [r for r in json.load(f) if r["split"] == args.split]
    if args.limit:
        records = sorted(records, key=lambda x: x["file"])[: args.limit]

    index = build_index(records, args.regions, args.stride)
    ipath = out / f"{args.split}_index.json"
    mpath = out / f"{args.split}.f16"
    nvec = index["vectors_per_position"]
    shape = (index["total_positions"], nvec, EMBED)

    nbytes = int(np.prod(shape)) * 2
    allocate(ipath, mpath, index, nbytes, args.rank)

    mm = np.memmap(mpath, dtype=np.float16, mode="r+", shape=shape)
    files = sorted(index["offsets"])[args.rank :: args.world]
    vit = load_backbone(device="cuda")
    wts = center_weights(device="cuda")

    loader = torch.utils.data.DataLoader(
        ClipFrames(root, files),
        batch_size=None,
        num_workers=args.workers,
        collate_fn=lambda x: x,
        prefetch_factor=4 if args.workers else None,
    )

    t0, done, nwin = time.time(), 0, 0
    for fname, frames in loader:
        n = frames.shape[0]
        off, length = index["offsets"][fname]
        if length == 0:
            done += 1
            continue
        frames = frames.cuda(non_blocking=True)
        starts = window_starts(n, args.stride)
        need = starts[-1] + WINDOW
        if need > n:                       # edge-repeat the final frames
            pad = frames[-1:].expand(need - n, -1, -1, -1)
            frames = torch.cat([frames, pad], dim=0)
        x = normalize(frames)              # (3, T, H, W)

        acc = torch.zeros(length, nvec, EMBED, device="cuda", dtype=torch.float32)
        wsum = torch.zeros(length, device="cuda", dtype=torch.float32)
        for b0 in range(0, len(starts), args.batch_windows):
            chunk = starts[b0 : b0 + args.batch_windows]
            batch = torch.stack([x[:, s : s + WINDOW] for s in chunk])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = window_tokens(vit, batch)
                reg = pool_regions(tok, args.regions)
            reg = reg.float()
            nwin += len(chunk)
            for k, s in enumerate(chunk):
                p0 = s // TUBELET
                hi = min(p0 + WINDOW // TUBELET, length)
                if hi <= p0:
                    continue
                m = hi - p0
                acc[p0:hi] += reg[k, :m] * wts[:m, None, None]
                wsum[p0:hi] += wts[:m]
        feats = (acc / wsum.clamp_min(1e-6)[:, None, None]).to(torch.float16)
        mm[off : off + length] = feats.cpu().numpy()
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            rate = done / el
            print(f"[rank{args.rank}] {done}/{len(files)} clips  {rate:.1f} clip/s  "
                  f"{nwin/el:.0f} win/s  eta {(len(files)-done)/rate/60:.1f} min", flush=True)
    mm.flush()
    print(f"[rank{args.rank}] DONE {done} clips in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
