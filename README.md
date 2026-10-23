# AVIGIL

Multimodal audio-visual forgery localization. Given a video, AVIGIL predicts

* **when** it was manipulated — intervals in seconds,
* **which modality** was manipulated — real / visual-only / audio-only / both,
* and **per-modality intervals** from the audio and visual branches separately.

The unit of prediction is a time interval, not a label.

---

## Install

```bash
git clone git@github.com:Helixometry-DASH/AVIGIL.git
cd AVIGIL

python -m venv .venv && source .venv/bin/activate

# install torch first, matched to your CUDA version
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install -e .
```

Python 3.10–3.12. A GPU is required for training; inference runs on CPU.

The visual backbone (`OpenGVLab/VideoMAEv2-Base`) is fetched from Hugging Face
on first use and kept frozen. Set `AVIGIL_BACKBONE` to a local snapshot
directory to run with no network access.

---

## Train setup

### 1. Arrange your data

Point `root` at a directory holding one folder per split plus a metadata JSON:

```
<root>/
├── metadata.min.json
├── train/
├── dev/
└── test/
```

`metadata.min.json` is a flat list of records, one per clip:

```json
[
  {
    "file": "train/000123.mp4",
    "split": "train",
    "video_frames": 116,
    "fake_periods": [[2.500, 3.224]],
    "modify_audio": true,
    "modify_video": true,
    "original": "train/000004.mp4"
  }
]
```

| field | type | meaning |
|---|---|---|
| `file` | str | clip path, relative to `root` |
| `split` | str | which split this clip belongs to |
| `video_frames` | int | number of video frames, and the authoritative frame count |
| `fake_periods` | `[[start, end]]` | manipulated intervals in seconds; `[]` if genuine |
| `modify_audio` | bool | whether the audio was manipulated |
| `modify_video` | bool | whether the video was manipulated |
| `original` | str \| null | the genuine clip this one was derived from; `null` if genuine |

Three label views are derived from each record, which is what keeps each branch
supervised on its own modality:

```
audio  = fake_periods if modify_audio else []
visual = fake_periods if modify_video else []
joint  = fake_periods if (modify_audio or modify_video) else []
```

So a clip whose audio alone was manipulated gets an empty visual target even
though `fake_periods` is non-empty. `original` groups each genuine clip with
every fake derived from it, so a regrouped split cannot leak between train and
test.

Clips are expected at 224×224, 25 fps, 16 kHz mono audio, each file carrying
both streams. Adjust `fps`, `sample_rate` and `image_size` in
`configs/data/dataset.yaml` if yours differ.

### 2. Cache the visual features

The visual backbone is frozen, so it runs once offline and training reads the
cache. One fp16 vector set per 80 ms tubelet: channel 0 the mean over all patch
tokens, channels 1: a 4×4 region grid.

```bash
for split in train dev test; do
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$r python scripts/extract_features.py \
      --root /path/to/dataset \
      --out  /path/to/features \
      --split $split --rank $r --world 4 &
  done
  wait
done
```

`--rank`/`--world` shards write disjoint byte ranges of the same file, so the
processes need no coordination. Per split:

```
<out>/<split>.f16          (total_positions, 17, 768)
<out>/<split>_index.json   file -> (offset, length) + build settings
```

Size is about 0.75 MB per second of video at `--regions 4`.

### 3. Train

```bash
python scripts/train.py configs/tasks/av_localization/m5.yaml \
  data.root=/path/to/dataset \
  data.feature_root=/path/to/features \
  --test
```

Everything trains in one pass — audio branch, visual branch and fusion head —
under the combined objective

```
L = L_joint + w_audio * L_audio + w_video * L_video + w_modality * CE
```

Per-modality terms matter: training on the joint label alone would supervise
the audio branch to find manipulations that leave no acoustic trace.

Overrides use OmegaConf dotlist syntax and win over the config files:

```bash
python scripts/train.py configs/tasks/av_localization/m5.yaml \
  train.epochs=8 data.loader.batch_size=32 log.name=av_localization/m5_e8
```

Results land in `experiments/<log.name>/<timestamp>/`:

```
config.yaml          the fully resolved config for this run
metrics.jsonl        per-epoch metric history
train.log
checkpoints/         best.pt, last.pt, epochNNN.pt
test_metrics.json    with --test
```

Variants, each adding one component to the previous:

| config | fusion |
|---|---|
| `m3.yaml` | aligned features concatenated, then a temporal conv trunk |
| `m4.yaml` | m3 + bidirectional cross-modal attention |
| `m5.yaml` | m4 + reliability gating on the cross-modal messages |

Reference run for m5: 4 epochs at batch size 16, about 1.25 h per epoch on one
A100-40GB.

### Useful knobs

| key | default | effect |
|---|---|---|
| `task.crop_seconds` | 4.0 | window both modalities are cropped to |
| `task.train_window_hop` | 2.0 | hop between enumerated training windows |
| `task.grid_block` | 4 | encoder tap the joint head predicts on (15.19 ms) |
| `task.video_upsample` | 2 | visual output grid (2 → 40 ms) |
| `loss.w_modality` | 0.5 | weight on the 4-way attribution head |
| `eval.monitor` | `AP@0.95` | metric that selects `best.pt` |

---

## Test setup

### Evaluate a trained run

```bash
python scripts/evaluate.py experiments/av_localization/m5/<timestamp> \
  --ckpt best.pt --split test --out test_metrics.json
```

Evaluation slides the 4.0375 s audio encoder window over each recording,
reassembles the features on the recording's own frame grid, and fuses once over
that whole timeline, so nothing is truncated. Boundaries are read as the
probability-weighted centroid of the boundary head.

| flag | default | meaning |
|---|---|---|
| `--ckpt` | `last.pt` | `best.pt`, `last.pt` or `epochNNN.pt` |
| `--crop` | `full` | `full` scores whole clips; `head`/`center` score one window |
| `--refine` | `soft` | `soft` continuous boundaries, `argmax` grid points |
| `--split` | `test` | which split to score |

Reported per run: AP at IoU 0.5 / 0.75 / 0.9 / 0.95, AR@N, boundary MAE,
clip EER and AUC, and modality accuracy overall and per class.

### Released checkpoint

`weights/avigil_m5/` holds the trained m5 checkpoint — 20.4 MB, 5,318,293
parameters — and the config needed to run it. Its own evaluation run:

| metric | value |
|---|---|
| AP@0.5 | 0.9423 |
| AP@0.75 | 0.9153 |
| AP@0.9 | 0.8864 |
| AP@0.95 | 0.8560 |
| AR@10 | 0.9317 |
| boundary MAE | 5.99 ms |
| clip EER | 0.0020 |
| clip AUC | 0.9999 |
| modality accuracy | 0.9373 |

Full output in `weights/avigil_m5/test_metrics.json`. These are the numbers for
the data that checkpoint was trained on; evaluate on your own data to get
numbers for it.

### Run on videos

```bash
python scripts/predict.py clip.mp4  --model weights/avigil_m5
python scripts/predict.py clips/    --model weights/avigil_m5 --out results.json
python scripts/predict.py clip.mp4 --json
python scripts/predict.py clip.mp4 --device cpu
```

Accepts `.mp4 .mov .mkv .avi .webm`. Each file needs both an audio and a video
stream. Input is resized to 224×224 and resampled to 25 fps and 16 kHz mono
automatically, and the report marks any file that was not already in that
format.

```
  FORGERY DETECTED   (clip probability 0.9999)
  modality  visual_only   real=0.000  visual_only=1.000  audio_only=0.000  both=0.000

  JOINT  (audio or video forged)
     #      start        end    length     conf
     1     2.500s     2.780s    0.280s   0.9994
```

`JOINT` is the answer; the per-branch tables below it show what each modality
contributed. Set `AVIGIL_DATASET_ROOT` to have clips from a dataset root
reported next to their ground-truth labels.

### Tests

```bash
pytest tests/
ruff check src/ scripts/ tests/
```

The grid-arithmetic and model tests need no data. The decoder-synchronisation
tests run only when `AVIGIL_DATASET_ROOT` is set, and skip otherwise.

---

## Layout

```
configs/
  base.yaml                       defaults merged under every config
  data/dataset.yaml               dataset paths and media constants
  tasks/av_localization/          _task.yaml + m3/m4/m5
src/avigil/
  data/                           dataset, label views, decoding, batching
  engine/                         trainer, sliding-window inference
  losses/                         localization losses
  metrics/                        AP@IoU, AR@N, EER, AUC
  models/
    audio/                        audio branch: layers, encoder, graph, decoder
    video/                        visual branch over cached features
    fusion/                       joint model and grid alignment
    backbone/                     VideoMAE V2 token extraction
  tasks/                          the training/evaluation task
  utils/                          config, logging, seeding
scripts/
  extract_features.py             cache visual features (run once)
  train.py                        train
  evaluate.py                     evaluate a finished run
  predict.py                      run on arbitrary videos
weights/avigil_m5/                released checkpoint + inference config
tests/
```

### Grids

The two branches run at different rates and neither divides the other, so
features are aligned by timestamp rather than by array index:

```
audio   hop 243 samples @ 16 kHz = 15.1875 ms = 65.84 Hz
video   hop 2 frames    @ 25 fps = 40.0 ms    = 25.00 Hz
```

Fusion happens on the finer audio grid, with the visual sequence interpolated
up to meet it, so the joint head can express a boundary to 15.19 ms.

### Environment variables

| variable | effect |
|---|---|
| `AVIGIL_BACKBONE` | path or Hugging Face id of the VideoMAE V2 weights |
| `AVIGIL_DATASET_ROOT` | dataset root(s) for ground-truth reporting and the sync tests |
