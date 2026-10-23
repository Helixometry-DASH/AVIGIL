"""AVIGIL -- multimodal audio-visual deepfake localization.

Given a video, answer three questions:

  1. is anything forged, and *when*  -- forged intervals in seconds
  2. which modality was forged       -- real / visual-only / audio-only / both
  3. per modality, when              -- separate audio and video intervals

Not a classifier. The unit of prediction is a time interval, not a label.
"""

__version__ = "0.1.0"
