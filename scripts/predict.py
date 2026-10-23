#!/usr/bin/env python
"""Localize audio-visual forgery in a video file or a folder of them.

    python scripts/predict.py clip.mp4 --model weights/avigil_m5
    python scripts/predict.py clips/ --out results.json

Runs the whole chain an input file needs: decode, resample to the grid the
model was trained on, VideoMAE V2 features, then windowed inference over the
recording's own timeline.

Inputs are resized to 224x224 and resampled to 25 fps and 16 kHz mono. The
report marks any file that was not already in that format, and confidences are
calibrated only for the training domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from avigil.models import build
from avigil.models.audio.grid import frame_to_time
from avigil.models.audio.proposals import decode_proposals
from avigil.models.backbone import (
    TUBELET,
    WINDOW,
    center_weights,
    load_backbone,
    normalize,
    pool_regions,
    window_starts,
    window_tokens,
)
from avigil.models.fusion.windowed_av import predict_av_full

FPS = 25.0
SIZE = 224
SR = 16000
MODALITY = ("real", "visual_only", "audio_only", "both")


# Set AVIGIL_DATASET_ROOT to a dataset root holding metadata.min.json and
# clips from it are reported alongside their ground-truth labels. Colon-separate
# several roots. Unset for arbitrary input.
import os  # noqa: E402

LAVDF_ROOTS = tuple(
    r for r in os.environ.get("AVIGIL_DATASET_ROOT", "").split(os.pathsep) if r
)
_META: dict | None = None


def ground_truth(path: str) -> dict | None:
    """The metadata entry for a clip under a configured dataset root, if any.

    Matched on the path *relative to a dataset root*, never on the bare
    filename: ``train/000001.mp4`` and ``test/000001.mp4`` both exist, so a
    basename lookup would silently return the wrong split's label.
    """
    global _META
    rel = None
    ap = str(Path(path).resolve())
    for root in LAVDF_ROOTS:
        if ap.startswith(root + "/"):
            rel = ap[len(root) + 1:]
            break
    if rel is None:
        return None
    if _META is None:
        _META = {}
        for root in LAVDF_ROOTS:
            f = Path(root) / "metadata.min.json"
            if f.exists():
                for r in json.loads(f.read_text()):
                    _META.setdefault(r["file"], r)
    r = _META.get(rel)
    if r is None:
        return None
    cls = ("real", "visual_only", "audio_only", "both")[
        (1 if r["modify_video"] else 0) + (2 if r["modify_audio"] else 0)]
    return {
        "class": cls, "split": r.get("split"),
        "modify_audio": bool(r["modify_audio"]),
        "modify_video": bool(r["modify_video"]),
        "intervals": [[round(float(a), 3), round(float(b), 3)] for a, b in r["fake_periods"]],
    }


def decode_media(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (frames uint8 (T,224,224,3) at 25 fps, audio float32 mono 16 kHz, info)."""
    import av

    info: dict = {}
    with av.open(path) as c:
        if not c.streams.video:
            raise RuntimeError(f"{path}: no video stream")
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        info["source_fps"] = float(v.average_rate) if v.average_rate else None
        info["source_size"] = [v.codec_context.width, v.codec_context.height]
        frames = [f.to_ndarray(format="rgb24", width=SIZE, height=SIZE)
                  for f in c.decode(video=0)]
    vid = np.stack(frames) if frames else np.zeros((0, SIZE, SIZE, 3), np.uint8)

    with av.open(path) as c:
        if not c.streams.audio:
            raise RuntimeError(f"{path}: no audio stream")
        a = c.streams.audio[0]
        info["source_sr"] = a.rate
        res = av.AudioResampler(format="fltp", layout="mono", rate=SR)
        chunks = []
        for fr in c.decode(audio=0):
            for out in res.resample(fr):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in res.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)

    # resample video to exactly 25 fps by nearest-timestamp pick
    src_fps = info["source_fps"] or FPS
    if vid.shape[0] and abs(src_fps - FPS) > 1e-3:
        dur = vid.shape[0] / src_fps
        idx = np.clip((np.arange(int(dur * FPS)) / FPS * src_fps).round().astype(int),
                      0, vid.shape[0] - 1)
        vid = vid[idx]
        info["resampled_video"] = True
    info["duration_s"] = round(max(vid.shape[0] / FPS, len(wav) / SR), 3)
    info["n_frames"] = int(vid.shape[0])
    return vid, wav, info


@torch.no_grad()
def video_features(vit, frames: np.ndarray, regions: int, device: str,
                   stride: int = 8) -> torch.Tensor:
    """(T,224,224,3) uint8 -> (1, P, regions**2+1, 768), P = T // TUBELET."""
    total = max(frames.shape[0] // TUBELET, 1)
    t = torch.from_numpy(frames).to(device)
    starts = window_starts(frames.shape[0], stride)
    need = starts[-1] + WINDOW
    if need > t.shape[0]:
        t = torch.cat([t, t[-1:].expand(need - t.shape[0], -1, -1, -1)], 0)
    x = normalize(t)
    nvec = regions * regions + 1
    acc = torch.zeros(total, nvec, 768, device=device)
    wsum = torch.zeros(total, device=device)
    wts = center_weights(device=device)
    for i in range(0, len(starts), 16):
        chunk = starts[i : i + 16]
        batch = torch.stack([x[:, s : s + WINDOW] for s in chunk])
        with torch.autocast(device, dtype=torch.bfloat16):
            reg = pool_regions(window_tokens(vit, batch), regions).float()
        for b, s in enumerate(chunk):
            p0 = s // TUBELET
            hi = min(p0 + WINDOW // TUBELET, total)
            if hi <= p0:
                continue
            m = hi - p0
            acc[p0:hi] += reg[b, :m] * wts[:m, None, None]
            wsum[p0:hi] += wts[:m]
    return (acc / wsum.clamp_min(1e-6)[:, None, None]).unsqueeze(0)


def load_model(bundle: Path, device: str):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(bundle / "config.yaml")
    m = OmegaConf.to_container(cfg.model, resolve=True)
    name = m.pop("name")
    a, v = m.pop("audio"), m.pop("video")
    a.pop("ckpt", None)
    v.pop("ckpt", None)
    model = build(name, audio_branch=build(a.pop("name"), **a),
                  video_branch=build(v.pop("name"), **v), **m)
    st = torch.load(bundle / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    return model.to(device).eval(), cfg


def build_engine(bundle: Path, device: str = "cuda") -> dict:
    """Load the model and backbone once. Loading costs ~15 s, so a directory
    scan must not pay it per file."""
    model, cfg = load_model(bundle, device)
    return {"model": model, "cfg": cfg, "device": device,
            "vit": load_backbone(device=device),
            "regions": int(cfg.model.get("video_regions", 2))}


def analyse(path: str, bundle: Path, device: str = "cuda", engine: dict | None = None) -> dict:
    eng = engine or build_engine(bundle, device)
    model, cfg, device = eng["model"], eng["cfg"], eng["device"]
    vid, wav, info = decode_media(path)
    feats = video_features(eng["vit"], vid, eng["regions"], device)
    audio = torch.from_numpy(wav)[None].to(device)
    mask = torch.ones(1, feats.shape[1], dtype=torch.bool, device=device)

    # the audio encoder window. The config states the crop in seconds, so fall
    # back to that, and finally to the 4.0375 s default.
    window = int(cfg.task.get("clip_samples", 0)) or int(
        round(float(cfg.task.get("crop_seconds", 4.0375)) * SR))
    r = predict_av_full(model, audio, feats, mask, block=4,
                        window=window, hop_frames=132)
    segs, scores = decode_proposals(
        r["frame_logits"], r["start_logits"], r["end_logits"],
        to_seconds=lambda f: frame_to_time(f, 4), mode="region", refine="soft",
        frame_threshold=float(cfg.eval.frame_threshold),
        peak_threshold=float(cfg.eval.peak_threshold),
        min_seconds=float(cfg.eval.min_segment_seconds),
        max_seconds=float(cfg.eval.max_segment_seconds),
        nms_iou=float(cfg.eval.nms_iou),
    )
    def props(fp, sp, ep, to_sec):
        sg, sc = decode_proposals(
            fp, sp, ep, to_seconds=to_sec, mode="region", refine="soft",
            frame_threshold=float(cfg.eval.frame_threshold),
            peak_threshold=float(cfg.eval.peak_threshold),
            min_seconds=float(cfg.eval.min_segment_seconds),
            max_seconds=float(cfg.eval.max_segment_seconds),
            nms_iou=float(cfg.eval.nms_iou),
        )
        return [{"start": round(float(a), 3), "end": round(float(b), 3),
                 "confidence": round(float(q), 4)}
                for (a, b), q in zip(sg, sc, strict=True)]

    # each branch on its own grid: audio at 15.19 ms, video at 40 ms
    up = int(cfg.task.get("video_upsample", 2))
    ab, vb = r["audio_branch"], r["video_branch"]
    audio_segs = props(ab["frame_logits"], ab["start_logits"], ab["end_logits"],
                       lambda f: frame_to_time(f, 4))
    video_segs = props(vb["frame_logits"], vb["start_logits"], vb["end_logits"],
                       lambda f, _u=up: f * (2.0 / 25.0) / _u)

    probs = np.exp(r["modality"] - r["modality"].max())
    probs = probs / probs.sum()
    step = frame_to_time(1, 4) - frame_to_time(0, 4)
    return {
        "file": Path(path).name,
        "ground_truth": ground_truth(path),
        "info": info,
        "clip_fake_probability": round(float(r["clip"]), 4),
        "modality": MODALITY[int(np.argmax(r["modality"]))],
        "modality_probabilities": {k: round(float(p), 4)
                                   for k, p in zip(MODALITY, probs, strict=True)},
        "segments": [
            {"start": round(float(a), 3), "end": round(float(b), 3),
             "confidence": round(float(q), 4)}
            for (a, b), q in zip(segs, scores, strict=True)
        ],
        "audio": {"fake_probability": round(float(r["audio_clip"]), 4),
                  "segments": audio_segs},
        "video": {"fake_probability": round(float(r["video_clip"]), 4),
                  "segments": video_segs},
        "frame_probability": [round(float(x), 4) for x in r["frame_logits"]],
        "frame_step_s": round(step, 6),
        "n_audio_windows": int(r["n_windows"]),
    }


def report(res: dict) -> None:
    """Human-readable summary for one file."""
    p = res["clip_fake_probability"]
    verdict = "FORGERY DETECTED" if p >= 0.5 else "no forgery detected"
    i = res["info"]
    print(f"\n  file      {res['file']}")
    print(f"  duration  {i['duration_s']} s   {i['n_frames']} frames @25fps   "
          f"{res['n_audio_windows']} audio windows")
    src = f"{i['source_size'][0]}x{i['source_size'][1]}"
    note = "" if (src == "224x224" and i.get("source_fps") == 25.0
                  and i.get("source_sr") == 16000) else "   [resampled -- outside training domain]"
    print(f"  source    {src}, {i.get('source_fps')} fps, {i.get('source_sr')} Hz{note}")
    print(f"\n  {verdict}   (clip probability {p:.4f})")
    print(f"  modality  {res['modality']}   "
          + "  ".join(f"{k}={v:.4f}" for k, v in res["modality_probabilities"].items()))
    def table(title, prob, segs):
        head = f"  {title}" + (f"   p={prob:.4f}" if prob is not None else "")
        print(f"\n{head}")
        if not segs:
            print("    none")
            return
        print(f"    {'#':>2}  {'start':>9s}  {'end':>9s}  {'length':>8s}  {'conf':>7s}")
        for n, s_ in enumerate(segs, 1):
            print(f"    {n:2d}  {s_['start']:8.3f}s  {s_['end']:8.3f}s  "
                  f"{s_['end'] - s_['start']:7.3f}s  {s_['confidence']:7.4f}")

    gt = res.get("ground_truth")
    if gt:
        iv = ", ".join(f"{a:.3f}-{b:.3f}" for a, b in gt["intervals"]) or "none"
        ok = "MATCH" if gt["class"] == res["modality"] else "MISMATCH"
        print(f"\n  GROUND TRUTH ({gt['split']} split)")
        print(f"    class     {gt['class']}   -> predicted {res['modality']}   [{ok}]")
        print(f"    intervals {iv}")
        print(f"    audio should fire: {'yes' if gt['modify_audio'] else 'NO'}"
              f"   video should fire: {'yes' if gt['modify_video'] else 'NO'}")
    # no labels available for this file: nothing to compare against

    table("JOINT   union of the audio and visual intervals", None, res["segments"])
    for name, key, flag in (("AUDIO   where in the audio, 15.19 ms grid", "audio", "modify_audio"),
                            ("VIDEO   where in the video, 40 ms grid", "video", "modify_video")):
        b = res[key]
        title = name
        if gt:
            fired = len(b["segments"]) > 0
            want = gt[flag]
            title += "   " + ("correct" if fired == want else
                              ("MISSED (should have fired)" if want else
                               "FALSE ALARM (should be silent)"))
        table(title, b["fake_probability"], b["segments"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a video file, or a directory of videos")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent
                                           / "weights/avigil_m5"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="", help="write JSON here")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    ap.add_argument("--limit", type=int, default=0, help="max files in directory mode")
    args = ap.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"error: no such path: {target}", file=sys.stderr)
        return 2
    files = (sorted(p for p in target.rglob("*") if p.suffix.lower()
                    in {".mp4", ".mov", ".mkv", ".avi", ".webm"})
             if target.is_dir() else [target])
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"error: no video files under {target}", file=sys.stderr)
        return 2

    eng = build_engine(Path(args.model), args.device)
    out, n_fake = [], 0
    for k, f in enumerate(files, 1):
        try:
            res = analyse(str(f), Path(args.model), args.device, engine=eng)
        except Exception as e:
            print(f"  [{k}/{len(files)}] {f.name}: FAILED -- {e}", file=sys.stderr)
            continue
        out.append(res)
        n_fake += res["clip_fake_probability"] >= 0.5
        if args.json:
            print(json.dumps({k2: v for k2, v in res.items()
                              if k2 != "frame_probability"}, indent=2))
        elif len(files) == 1:
            report(res)
        else:
            seg = ", ".join(f"{s['start']:.2f}-{s['end']:.2f}s" for s in res["segments"])
            print(f"  [{k}/{len(files)}] {f.name:24s} p={res['clip_fake_probability']:.4f} "
                  f"{res['modality']:12s} {seg or '-'}")
    if len(files) > 1:
        print(f"\n  {len(out)} analysed, {n_fake} flagged as forged, "
              f"{len(out) - n_fake} clean")
    if args.out:
        Path(args.out).write_text(json.dumps(out if len(out) != 1 else out[0], indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
