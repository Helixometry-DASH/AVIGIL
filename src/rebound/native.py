"""Attributed AVIGIL experts with explicit masks and continuous offsets."""
import importlib
import json
import sys
from pathlib import Path
import torch
from torch import nn
from .io import ROOT, sha256


def import_vendor():
    root = ROOT / 'third_party' / 'AVIGIL'
    provenance = json.loads((root / 'PROVENANCE.json').read_text())
    for file, digest in provenance['files_sha256'].items():
        if sha256(root / file) != digest:
            raise ValueError('Audited source dependency changed: ' + file)
    source = (root / 'src').resolve()
    if 'avigil' in sys.modules:
        existing = Path(sys.modules['avigil'].__file__).resolve()
        if source not in existing.parents:
            raise RuntimeError('Another AVIGIL import is active; use an isolated ReBound process')
    sys.path.insert(0, str(source))
    return provenance


class NativeExperts(nn.Module):
    def __init__(self, source='audio_source'):
        super().__init__()
        if source not in ('audio_source', 'visual_source', 'mixed_source'):
            raise ValueError('ReBound experts require an explicit source protocol')
        import_vendor()
        audio = importlib.import_module('avigil.models.audio.localizer').AudioLocalizer
        video = importlib.import_module('avigil.models.video.model').VideoLocalizer
        joint = importlib.import_module('avigil.models.fusion.model').AVIGIL
        self.core = joint(audio(variant='a3'), video(variant='v3'), freeze_branches=False)
        # The attributed a3 forward never consumes these allocated parameters:
        # residual bn1 is bypassed, and the graph master output is discarded.
        # Freeze them explicitly so DDP does not wait for nonexistent gradients.
        for name, parameter in self.core.named_parameters():
            if (name.startswith('audio_branch.encoder.blocks.') and '.bn1.' in name) or (
                name.startswith('audio_branch.hsgal.') and any(token in name for token in
                    ('.master', '.att_weightM', '.att_projM.', '.proj_with_attM.', '.proj_without_attM.'))):
                parameter.requires_grad_(False)
        self.source = source
        self.offset_av = nn.Linear(256, 2)
        self.offset_a = nn.Linear(128, 2)
        self.offset_v = nn.Linear(256, 2)
        for layer in (self.offset_av, self.offset_a, self.offset_v):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def attach(head, features, times, mask, offsets, offset_range=3.):
        delta = offsets(features.transpose(1, 2)).float().tanh() * offset_range
        step = times[1] - times[0]
        return {**head, 'features': features.transpose(1, 2), 'times': times[None].expand(len(features), -1),
                'valid': mask, 'start_offset': delta[..., 0], 'end_offset': delta[..., 1],
                'start_time': times[None] + step * delta[..., 0], 'end_time': times[None] + step * delta[..., 1]}

    def forward(self, audio, video, video_valid, audio_length, video_frames):
        core = self.core
        n = (audio_length - 128).clamp_min(0) // 243
        ta = (audio.shape[-1] - 128) // 243
        amask = torch.arange(ta, device=audio.device)[None] < n[:, None]
        # Graph attention in the attributed audio branch has no node mask.
        # Crop each length bucket before its encoder/graph, then pad outputs;
        # otherwise zero-padded coarse nodes still become graph keys.
        pieces = [None] * len(audio)
        for length in torch.unique(audio_length).tolist():
            if length < 128 + 2187:
                raise ValueError('Audio crop is too short for the coarse native encoder')
            positions = torch.nonzero(audio_length == length, as_tuple=False).flatten()
            count = (length - 128) // 243
            masks = torch.ones(len(positions), count, device=audio.device, dtype=torch.bool)
            value = core.audio_branch(audio[positions, :length], frame_mask=masks,
                                      lengths=audio_length[positions])
            for local, original in enumerate(positions.tolist()):
                pieces[original] = {k: torch.nn.functional.pad(v[local], (0, ta - count))
                                    if k != 'clip_logit' else v[local] for k, v in value.items()}
        oa = {key: torch.stack([piece[key] for piece in pieces]) for key in pieces[0]}
        ov = core.video_branch(video, video_valid)
        vmask = torch.arange(ov['frame_logits'].shape[-1], device=audio.device)[None] < video_frames[:, None]
        audio_times = (torch.arange(ta, device=audio.device).float() * 243 + 64) / 16000
        video_times = (torch.arange(ov['frame_logits'].shape[-1], device=audio.device).float() + .5) / 25
        va = torch.cat((ov['features'], torch.stack([ov[k] for k in ('frame_logits', 'start_logits', 'end_logits')], 1)), 1)
        aa = torch.cat((oa['features'], torch.stack([oa[k] for k in ('frame_logits', 'start_logits', 'end_logits')], 1)), 1)
        resample = importlib.import_module('avigil.models.fusion.align').resample_to_grid
        va = resample(va, video_times, audio_times, vmask)
        ha = core.norm_a(core.proj_a(aa)).transpose(1, 2) * amask[..., None]
        hv = core.norm_v(core.proj_v(va)).transpose(1, 2) * amask[..., None]
        old_ha = ha
        ha = core.a2v(ha, hv, amask) * amask[..., None]
        hv = core.v2a(hv, old_ha, amask) * amask[..., None]
        x = core.fuse(torch.cat((ha, hv), -1).transpose(1, 2)) * amask[:, None]
        for block in core.trunk:
            x = block(x) * amask[:, None]
        pooled = x.sum(-1) / amask.sum(-1).clamp_min(1)[:, None]
        head = {'frame_logits': core.head_frame(x).squeeze(1), 'start_logits': core.head_start(x).squeeze(1),
                'end_logits': core.head_end(x).squeeze(1), 'clip_logit': core.head_clip(pooled).squeeze(-1),
                'modality_logits': core.head_modality(pooled)}
        result = self.attach(head, x, audio_times, amask, self.offset_av)
        result['audio'] = self.attach(oa, oa['features'], audio_times, amask, self.offset_a)
        result['video'] = self.attach(ov, ov['features'], video_times, vmask, self.offset_v)
        return result


def forbid_mixed_initialization(checkpoint_metadata, source):
    if checkpoint_metadata.get('source') != source or checkpoint_metadata.get('stage') != 'experts':
        raise ValueError('Mixed, other-source or unknown-provenance expert checkpoint is prohibited')
