"""Model registry and the three localizer families.

    audio/      AudioLocalizer  -- aloc_a0 .. aloc_a3
    video/      VideoLocalizer  -- vloc_v0 .. vloc_v3
    fusion/     AVIGIL          -- avigil_m3 .. avigil_m5
    backbone/   the VideoMAE V2 token tap (not a registered model)

Importing this module is what registers every name, so ``build("avigil_m5")``
works without the caller knowing which file it lives in.
"""

from .audio import AudioLocalizer
from .base import AVLocalizer
from .fusion import AVIGIL
from .registry import available, build, register
from .video import VideoLocalizer

__all__ = [
    "AVIGIL",
    "AVLocalizer",
    "AudioLocalizer",
    "VideoLocalizer",
    "available",
    "build",
    "register",
]
