"""Video/audio decoding.

Kept behind one function so the backend can be swapped without touching the
Dataset. PyAV is the default: it reads the video and audio streams of one
container in a single pass.
"""

from __future__ import annotations

import numpy as np


def decode_clip(
    path: str,
    start_frame: int = 0,
    n_frames: int | None = None,
    fps: float = 25.0,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a frame window and the matching audio span.

    Returns ``(video, audio)`` where video is uint8 ``(T, H, W, 3)`` and audio
    is float32 ``(S,)`` mono, with ``S == round(T / fps * sample_rate)``.

    Audio is sliced by the *frame window*, not by the container duration, so
    the two modalities stay aligned -- the whole point of an AV model.
    """
    import av

    t0 = start_frame / fps
    with av.open(path) as container:
        vstream = container.streams.video[0]
        astream = container.streams.audio[0] if container.streams.audio else None

        vstream.thread_type = "AUTO"
        frames = []
        # seek slightly before the target, then drop frames until t0
        if start_frame > 0:
            container.seek(int(t0 / vstream.time_base), stream=vstream)
        for frame in container.decode(video=0):
            idx = round(float(frame.pts * vstream.time_base) * fps)
            if idx < start_frame:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if n_frames is not None and len(frames) >= n_frames:
                break
        video = (
            np.stack(frames)
            if frames
            else np.zeros((0, vstream.height, vstream.width, 3), dtype=np.uint8)
        )

        audio = np.zeros(0, dtype=np.float32)
        if astream is not None:
            container.seek(0)
            chunks = [
                f.to_ndarray().astype(np.float32).reshape(-1)
                for f in container.decode(audio=0)
            ]
            if chunks:
                full = np.concatenate(chunks)
                a0 = int(round(t0 * sample_rate))
                want = (
                    int(round(len(video) / fps * sample_rate))
                    if n_frames is None
                    else int(round(n_frames / fps * sample_rate))
                )
                audio = full[a0 : a0 + want]

    return video, audio


def decode_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Whole audio track as float32 mono in [-1, 1].

    The track is expected to be at ``sample_rate`` already; a mismatch raises
    rather than being silently resampled.

    **Audio/video synchronisation.** Sample 0 of the returned array is t=0, the
    same instant as video frame 0, so both modalities index onto one timeline.
    Note that the first audio *packet* carries ``pts=-1024`` (AAC-LC encoder
    priming, -64 ms); PyAV's decoder discards the priming and its first output
    frame reports ``pts=0``, so the decoded array needs no offset correction.

    ``len(wav)`` comes out 1024 samples short of ``duration * sample_rate``,
    which is end-of-stream rounding rather than a start offset.
    """
    import av

    with av.open(path) as container:
        if not container.streams.audio:
            raise RuntimeError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        if stream.rate != sample_rate:
            raise RuntimeError(
                f"{path}: audio is {stream.rate} Hz, expected {sample_rate} Hz"
            )
        chunks = [f.to_ndarray() for f in container.decode(audio=0)]

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    wav = np.concatenate([c.reshape(c.shape[0], -1).mean(axis=0) for c in chunks]) \
        if chunks[0].ndim == 2 and chunks[0].shape[0] > 1 \
        else np.concatenate([c.reshape(-1) for c in chunks])
    wav = wav.astype(np.float32)
    if np.issubdtype(chunks[0].dtype, np.integer):
        wav /= 32768.0
    return wav
