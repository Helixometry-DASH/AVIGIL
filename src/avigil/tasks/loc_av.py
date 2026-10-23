"""Joint audio-visual localization task.

Reports three things every epoch:

1. **Joint localization** -- AP at IoU 0.5 / 0.75 / 0.9 / 0.95 and AR@N
   against ``fake_periods`` wherever either modality is forged. The union of
   the two branches' proposals is scored alongside it for reference.
2. **Modality attribution** -- accuracy over real / visual-only / audio-only /
   both, overall and per class.
3. **Per-branch localization** -- each branch against its own modality's
   target, so the audio and visual heads stay measurable on their own.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data.av_loc import AVLocDataset, collate_av_loc
from ..losses.localization import LocalizationLoss
from ..metrics.classification import eer, roc_auc
from ..metrics.localization import (
    auprc,
    average_precision,
    boundary_error,
    evaluate_localization,
    frame_f1,
    nms,
)
from ..models.audio.grid import frame_to_time
from ..models.audio.proposals import decode_proposals
from ..models.registry import build as build_model
from .base import Task
from .registry import register_task

MODALITY_NAMES = ("real", "visual_only", "audio_only", "both")


class AVLocalizationTask(Task):
    name = "av_localization"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.block = int(cfg.task.get("grid_block", 4))
        self.w_modality = float(cfg.loss.get("w_modality", 0.5))
        self.w_audio = float(cfg.loss.get("w_audio", 0.0))
        self.w_video = float(cfg.loss.get("w_video", 0.0))
        self.loss_fn = LocalizationLoss(
            w_frame=cfg.loss.w_frame, w_dice=cfg.loss.w_dice,
            w_boundary=cfg.loss.w_boundary, w_clip=cfg.loss.w_clip,
            pos_weight=cfg.loss.get("pos_weight", None),
        )

    # ---- data -------------------------------------------------------------
    def build_dataset(self, split: str, train: bool) -> AVLocDataset:
        d = self.cfg.data
        return AVLocDataset(
            root=d.root, feature_root=d.feature_root, split=d.splits[split],
            metadata=d.metadata, sample_rate=d.sample_rate,
            crop_seconds=float(self.cfg.task.crop_seconds),
            crop=self.cfg.task.train_crop if train else self.cfg.task.eval_crop,
            block=self.block,
            video_upsample=int(self.cfg.task.get("video_upsample", 2)),
            video_regions=self.cfg.task.get("video_regions"),
            boundary_sigma=float(self.cfg.task.boundary_sigma),
            # training enumerates every window so nothing is skipped; evaluation
            # scores whole recordings, so it needs no window index
            window_hop=float(self.cfg.task.get("train_window_hop", 0.0)) if train else 0.0,
            seed=int(self.cfg.seed),
        )

    # ---- model ------------------------------------------------------------
    def build_model(self) -> torch.nn.Module:
        from omegaconf import OmegaConf

        m = OmegaConf.to_container(self.cfg.model, resolve=True)
        name = m.pop("name")
        a_cfg, v_cfg = m.pop("audio"), m.pop("video")
        a_ckpt, v_ckpt = a_cfg.pop("ckpt", None), v_cfg.pop("ckpt", None)
        audio = build_model(a_cfg.pop("name"), **a_cfg)
        video = build_model(v_cfg.pop("name"), **v_cfg)
        for branch, ckpt, tag in ((audio, a_ckpt, "audio"), (video, v_ckpt, "video")):
            if not ckpt:
                if m.get("freeze_branches", True):
                    raise ValueError(
                        f"model.{tag}.ckpt is required when freeze_branches=true: "
                        f"a frozen branch needs trained weights"
                    )
                continue                      # trained end to end from scratch
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            branch.load_state_dict(state["model"])
        return build_model(name, audio_branch=audio, video_branch=video, **m)

    # ---- optimisation -----------------------------------------------------
    def compute_loss(self, model, batch) -> tuple[torch.Tensor, dict[str, float]]:
        """The joint objective, trained in one pass.

            L = w_av * L_joint + w_a * L_audio + w_v * L_video + w_mod * CE

        Each branch keeps its **own** supervision, on its own grid and under
        its own mask: the audio branch is scored against the audio intervals
        and the video branch against the visual ones, which may lie over
        entirely different spans. Training on the joint view alone would teach
        the audio branch to find manipulations that leave no acoustic trace,
        so the per-modality terms are what make joint training correct rather
        than merely simultaneous.
        """
        pred = model(batch["audio"], batch["feats"], batch["video_mask"])
        tm = batch["tmask"].float()
        vm = batch["vmask_out"].float()

        def view(suffix, mask):
            sfx = f"_{suffix}" if suffix else ""
            return {
                "frame": batch[f"frame{sfx}"], "start": batch[f"start{sfx}"],
                "end": batch[f"end{sfx}"],
                "start_valid": batch[f"start_valid{sfx}"] * mask,
                "end_valid": batch[f"end_valid{sfx}"] * mask,
                "clip": batch["clip" + sfx], "frame_mask": mask,
            }

        loss, stats = self.loss_fn(pred, view("", tm))
        stats = {f"av_{k}": v for k, v in stats.items()}

        if self.w_audio > 0:
            la, sa = self.loss_fn(pred["audio"], view("a", tm))
            loss = loss + self.w_audio * la
            stats["a_loss"] = float(la)
        if self.w_video > 0:
            lv, sv = self.loss_fn(pred["video"], view("v", vm))
            loss = loss + self.w_video * lv
            stats["v_loss"] = float(lv)

        l_mod = F.cross_entropy(pred["modality_logits"], batch["modality"])
        loss = loss + self.w_modality * l_mod
        with torch.no_grad():
            stats["l_modality"] = float(l_mod)
            stats["loss"] = float(loss)
            p = torch.sigmoid(pred["frame_logits"].float())
            hit = ((p >= 0.5) == (batch["frame"] > 0.5)).float() * tm
            stats["frame_acc"] = float(hit.sum() / tm.sum().clamp_min(1.0))
            stats["mod_acc"] = float(
                (pred["modality_logits"].argmax(-1) == batch["modality"]).float().mean())
        return loss, stats

    @torch.no_grad()
    def _predict_windowed(self, model, batch, cid) -> dict[str, torch.Tensor]:
        """One recording at a time: slide the audio encoder, fuse over the whole
        timeline, then decode proposals once on the recording's own grid.

        The fusion is deliberately *not* windowed -- it is meant to see the
        assembled sequence spanning the recording, not a 4 s slice.
        """
        # imported here, not at module scope: windowed_av reaches into
        # engine.windowed, and engine/__init__ imports the trainer, which
        # imports this module -- a top-level import closes that cycle
        from ..models.fusion.windowed_av import assemble_audio, predict_av_full

        alen = (batch["length"] if "length" in batch else batch["audio_length"]).cpu().numpy()
        vlen = batch["video_length"].cpu().numpy()
        up = int(self.cfg.task.get("video_upsample", 2))
        win = int(self.cfg.task.get("clip_samples", 64600))
        hop = int(self.cfg.eval.get("window_hop_frames", 132))
        wb = int(self.cfg.eval.get("window_batch", 16))
        acc: dict[str, list] = {k: [] for k in
                                ("js", "jq", "jc", "us", "uq", "uc", "gs", "gc",
                                 "as_", "aq", "ac", "vs_", "vq", "vc",
                                 "gas", "gac", "gvs", "gvc")}
        cs, mp = [], []
        for i in range(len(cid)):
            a = batch["audio"][i : i + 1, : int(alen[i])]
            v = batch["feats"][i : i + 1, : int(vlen[i])]
            vmk = batch["video_mask"][i : i + 1, : int(vlen[i])]
            r = predict_av_full(model, a, v, vmk, block=self.block, window=win,
                                hop_frames=hop, batch_windows=wb)
            cs.append(r["clip"])
            mp.append(int(np.argmax(r["modality"])))
            seg, q = self._decode(r["frame_logits"], r["start_logits"], r["end_logits"],
                                  lambda f: frame_to_time(f, self.block))
            if len(seg):
                acc["js"].append(seg)
                acc["jq"].append(q)
                acc["jc"].append(np.full(len(seg), cid[i]))
            # the union baseline, from the same branches, same pass
            _, al = assemble_audio(model.audio_branch, a, self.block, win, hop, wb)
            ap = torch.sigmoid(al.float())[0].cpu().numpy()
            sa, qa = self._decode(ap[0], ap[1], ap[2],
                                  lambda f: frame_to_time(f, self.block))
            ov = model.video_branch(v, vmk)
            vp = [torch.sigmoid(ov[k].float())[0].cpu().numpy()
                  for k in ("frame_logits", "start_logits", "end_logits")]
            sv, qv = self._decode(vp[0], vp[1], vp[2],
                                  lambda f, _u=up: f * (2.0 / 25.0) / _u)
            su = np.concatenate([x for x in (sa, sv) if len(x)]) if (len(sa) or len(sv)) \
                else np.zeros((0, 2), np.float32)
            qu = np.concatenate([x for x in (qa, qv) if len(x)]) if (len(qa) or len(qv)) \
                else np.zeros(0, np.float32)
            if len(sa):
                acc["as_"].append(sa)
                acc["aq"].append(qa)
                acc["ac"].append(np.full(len(sa), cid[i]))
            if len(sv):
                acc["vs_"].append(sv)
                acc["vq"].append(qv)
                acc["vc"].append(np.full(len(sv), cid[i]))
            for key, seg_key, cid_key in (("seg_audio", "gas", "gac"),
                                          ("seg_visual", "gvs", "gvc")):
                gm = batch[key][i].cpu().numpy()
                if len(gm):
                    acc[seg_key].append(gm)
                    acc[cid_key].append(np.full(len(gm), cid[i]))
            if len(su):
                su, qu = nms(su.astype(np.float64), qu.astype(np.float64),
                             iou_threshold=float(self.cfg.eval.nms_iou))
                acc["us"].append(su)
                acc["uq"].append(qu)
                acc["uc"].append(np.full(len(su), cid[i]))
            g = batch["segments"][i].cpu().numpy()
            if len(g):
                acc["gs"].append(g)
                acc["gc"].append(np.full(len(g), cid[i]))
        return self._pack(acc, cs, mp, batch, cid)

    def _pack(self, acc, cs, mp, batch, cid) -> dict[str, torch.Tensor]:
        def cat2(v):
            return np.concatenate(v) if v else np.zeros((0, 2), np.float32)

        def cat1(v):
            return np.concatenate(v) if v else np.zeros(0, np.float32)

        t = torch.from_numpy
        return {
            "pred_seg": t(cat2(acc["js"])), "pred_score": t(cat1(acc["jq"])),
            "pred_cid": t(cat1(acc["jc"]).astype(np.float32)),
            "union_seg": t(cat2(acc["us"])), "union_score": t(cat1(acc["uq"])),
            "union_cid": t(cat1(acc["uc"]).astype(np.float32)),
            "gt_seg": t(cat2(acc["gs"])), "gt_cid": t(cat1(acc["gc"]).astype(np.float32)),
            "frame_prob": torch.zeros(0), "frame_tgt": torch.zeros(0),
            "clip_score": torch.tensor(cs, dtype=torch.float32),
            "clip_tgt": batch["clip"].float().cpu(),
            "mod_pred": torch.tensor(mp, dtype=torch.long),
            "mod_tgt": batch["modality"].cpu(),
            "clip_id": batch["index"].cpu().float(),
            "audio_seg": t(cat2(acc["as_"])), "audio_score": t(cat1(acc["aq"])),
            "audio_cid": t(cat1(acc["ac"]).astype(np.float32)),
            "video_seg": t(cat2(acc["vs_"])), "video_score": t(cat1(acc["vq"])),
            "video_cid": t(cat1(acc["vc"]).astype(np.float32)),
            "gt_audio_seg": t(cat2(acc["gas"])),
            "gt_audio_cid": t(cat1(acc["gac"]).astype(np.float32)),
            "gt_video_seg": t(cat2(acc["gvs"])),
            "gt_video_cid": t(cat1(acc["gvc"]).astype(np.float32)),
        }

    def _decode(self, fp, sp, ep, to_sec):
        e = self.cfg.eval
        return decode_proposals(
            fp, sp, ep, to_seconds=to_sec,
            mode=e.get("proposal_mode", "region"),
            refine=e.get("boundary_refine", "soft"),
            frame_threshold=float(e.frame_threshold), peak_threshold=float(e.peak_threshold),
            min_seconds=float(e.min_segment_seconds), max_seconds=float(e.max_segment_seconds),
            nms_iou=float(e.nms_iou),
        )

    @torch.no_grad()
    def predict(self, model, batch) -> dict[str, torch.Tensor]:
        cid = batch["index"].cpu().numpy()
        if bool(self.cfg.eval.get("windowed", False)):
            return self._predict_windowed(model, batch, cid)
        out = model(batch["audio"], batch["feats"], batch["video_mask"])
        sig = lambda x: torch.sigmoid(x.float()).cpu().numpy()  # noqa: E731
        jf, js, je = sig(out["frame_logits"]), sig(out["start_logits"]), sig(out["end_logits"])
        af, as_, ae = (sig(out["audio"][k]) for k in ("frame_logits", "start_logits", "end_logits"))
        vf, vs, ve = (sig(out["video"][k]) for k in ("frame_logits", "start_logits", "end_logits"))
        vlen = batch["video_length"].cpu().numpy()
        up = int(self.cfg.task.get("video_upsample", 2))
        a_sec = lambda f: frame_to_time(f, self.block)          # noqa: E731
        v_sec = lambda f: f * (2.0 / 25.0) / up                  # noqa: E731

        acc: dict[str, list] = {k: [] for k in
                                ("js", "jq", "jc", "us", "uq", "uc", "gs", "gc")}
        for i in range(len(cid)):
            seg, q = self._decode(jf[i], js[i], je[i], a_sec)
            if len(seg):
                acc["js"].append(seg)
                acc["jq"].append(q)
                acc["jc"].append(np.full(len(seg), cid[i]))
            # reference: union of the two branches' proposals, temporal NMS
            sa, qa = self._decode(af[i], as_[i], ae[i], a_sec)
            nv = int(vlen[i]) * up
            sv, qv = self._decode(vf[i, :nv], vs[i, :nv], ve[i, :nv], v_sec)
            su = np.concatenate([x for x in (sa, sv) if len(x)]) if (len(sa) or len(sv)) \
                else np.zeros((0, 2), np.float32)
            qu = np.concatenate([x for x in (qa, qv) if len(x)]) if (len(qa) or len(qv)) \
                else np.zeros(0, np.float32)
            if len(sa):
                acc["as_"].append(sa)
                acc["aq"].append(qa)
                acc["ac"].append(np.full(len(sa), cid[i]))
            if len(sv):
                acc["vs_"].append(sv)
                acc["vq"].append(qv)
                acc["vc"].append(np.full(len(sv), cid[i]))
            for key, seg_key, cid_key in (("seg_audio", "gas", "gac"),
                                          ("seg_visual", "gvs", "gvc")):
                gm = batch[key][i].cpu().numpy()
                if len(gm):
                    acc[seg_key].append(gm)
                    acc[cid_key].append(np.full(len(gm), cid[i]))
            if len(su):
                su, qu = nms(su.astype(np.float64), qu.astype(np.float64),
                             iou_threshold=float(self.cfg.eval.nms_iou))
                acc["us"].append(su)
                acc["uq"].append(qu)
                acc["uc"].append(np.full(len(su), cid[i]))
            g = batch["segments"][i].cpu().numpy()
            if len(g):
                acc["gs"].append(g)
                acc["gc"].append(np.full(len(g), cid[i]))

        cat2 = lambda v: np.concatenate(v) if v else np.zeros((0, 2), np.float32)  # noqa: E731
        cat1 = lambda v: np.concatenate(v) if v else np.zeros(0, np.float32)       # noqa: E731
        t = torch.from_numpy
        return {
            "pred_seg": t(cat2(acc["js"])), "pred_score": t(cat1(acc["jq"])),
            "pred_cid": t(cat1(acc["jc"]).astype(np.float32)),
            "union_seg": t(cat2(acc["us"])), "union_score": t(cat1(acc["uq"])),
            "union_cid": t(cat1(acc["uc"]).astype(np.float32)),
            "gt_seg": t(cat2(acc["gs"])), "gt_cid": t(cat1(acc["gc"]).astype(np.float32)),
            "frame_prob": t(jf.reshape(-1)), "frame_tgt": batch["frame"].reshape(-1).float().cpu(),
            "clip_score": torch.sigmoid(out["clip_logit"].float()).cpu(),
            "clip_tgt": batch["clip"].float().cpu(),
            "mod_pred": out["modality_logits"].argmax(-1).cpu(),
            "mod_tgt": batch["modality"].cpu(),
        }

    # ---- metrics ----------------------------------------------------------
    def evaluate(self, collected) -> dict[str, float]:
        def pack(seg_k, score_k, cid_k):
            c = np.asarray(collected[cid_k]).astype(np.int64)
            s = np.asarray(collected[seg_k]).reshape(-1, 2)
            q = np.asarray(collected[score_k]).reshape(-1)
            return [(int(c[i]), float(s[i, 0]), float(s[i, 1]), float(q[i])) for i in range(len(c))]

        gc = np.asarray(collected["gt_cid"]).astype(np.int64)
        gs = np.asarray(collected["gt_seg"]).reshape(-1, 2)
        gt = {int(c): gs[gc == c] for c in np.unique(gc)}
        joint = pack("pred_seg", "pred_score", "pred_cid")
        union = pack("union_seg", "union_score", "union_cid")

        out = evaluate_localization(
            joint, gt, ap_iou=tuple(self.cfg.eval.ap_iou),
            ar_topk=tuple(self.cfg.eval.ar_topk), ar_iou=tuple(self.cfg.eval.ar_iou))
        # windowed eval reassembles per recording and does not emit a flat
        # frame array, so the frame-level metrics are reported only when it did
        if np.asarray(collected["frame_prob"]).size:
            out.update(frame_f1(collected["frame_prob"], collected["frame_tgt"],
                                float(self.cfg.eval.frame_threshold)))
            out["frame_auprc"] = auprc(collected["frame_prob"], collected["frame_tgt"])
        out.update(boundary_error(joint, gt, iou_threshold=0.5))

        # the baseline the fusion has to beat, scored identically
        for thr in self.cfg.eval.ap_iou:
            out[f"union_AP@{thr}"] = average_precision(union, gt, iou_threshold=float(thr))

        cs, ct = np.asarray(collected["clip_score"]), np.asarray(collected["clip_tgt"])
        out["clip_eer"], _ = eer(cs, ct)
        out["clip_auc"] = roc_auc(cs, ct)

        mp, mt = np.asarray(collected["mod_pred"]), np.asarray(collected["mod_tgt"])
        out["mod_acc"] = float((mp == mt).mean())
        for k, nm in enumerate(MODALITY_NAMES):
            m = mt == k
            out[f"mod_acc/{nm}"] = float((mp[m] == mt[m]).mean()) if m.any() else float("nan")
        out.update(self._per_modality(collected, gt))
        out["n_proposals"] = float(len(joint))
        out["n_union_proposals"] = float(len(union))
        out["n_gt_segments"] = float(len(gs))
        return out

    def _per_modality(self, collected, gt) -> dict[str, float]:
        """Can the model tell WHICH modality was forged, and stay quiet on the other?

        Four cases -- real / visual-only /
        audio-only / both -- and the interesting ones are the two asymmetric
        ones. For an audio-only forgery the audio branch should fire and the
        video branch should stay silent; ungated cross-modal fusion is exactly
        what breaks that, which is the whole reason m5 has a reliability gate.

        So this reports, per modality class:
          ap_joint/<class>   joint localization AP@0.5 on that class
          leak/<class>       proposals per clip from the branch whose modality
                             is GENUINE -- a false-alarm rate, lower is better
        """
        g = lambda k: np.asarray(collected[k])                      # noqa: E731
        mt, cids = g("mod_tgt").astype(int), g("clip_id").astype(int)
        out: dict[str, float] = {}

        def pack(seg_k, score_k, cid_k):
            c = g(cid_k).astype(int)
            sg = g(seg_k).reshape(-1, 2)
            q = g(score_k).reshape(-1)
            return [(int(c[i]), float(sg[i, 0]), float(sg[i, 1]), float(q[i]))
                    for i in range(len(c))]

        joint = pack("pred_seg", "pred_score", "pred_cid")
        audio = pack("audio_seg", "audio_score", "audio_cid")
        video = pack("video_seg", "video_score", "video_cid")

        for k, name in enumerate(MODALITY_NAMES):
            ids = {int(c) for c in cids[mt == k]}
            if not ids:
                continue
            sub_gt = {c: v for c, v in gt.items() if c in ids}
            if sub_gt:
                out[f"ap_joint/{name}"] = average_precision(
                    [p for p in joint if p[0] in ids], sub_gt, iou_threshold=0.5)
            # the branch whose modality is genuine should produce nothing
            if name == "audio_only":          # video is genuine
                out["leak/video_on_audio_only"] = len(
                    [p for p in video if p[0] in ids]) / len(ids)
            elif name == "visual_only":       # audio is genuine
                out["leak/audio_on_visual_only"] = len(
                    [p for p in audio if p[0] in ids]) / len(ids)
            elif name == "real":              # both genuine
                out["leak/audio_on_real"] = len(
                    [p for p in audio if p[0] in ids]) / len(ids)
                out["leak/video_on_real"] = len(
                    [p for p in video if p[0] in ids]) / len(ids)

        # and each branch scored against its OWN modality's ground truth
        for tag, preds, gs, gc in (("audio", audio, "gt_audio_seg", "gt_audio_cid"),
                                   ("video", video, "gt_video_seg", "gt_video_cid")):
            cc = g(gc).astype(int)
            ss = g(gs).reshape(-1, 2)
            own = {int(c): ss[cc == c] for c in np.unique(cc)} if len(cc) else {}
            if own:
                out[f"ap_{tag}_own"] = average_precision(preds, own, iou_threshold=0.5)
        return out

    @property
    def monitor(self) -> tuple[str, str]:
        return (str(self.cfg.eval.get("monitor", "AP@0.95")), "max")

    @property
    def collate(self):
        return collate_av_loc

    @property
    def collate_pads(self) -> bool:
        return True


@register_task("av_localization")
def build_t6(cfg) -> AVLocalizationTask:
    return AVLocalizationTask(cfg)
