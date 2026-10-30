# ReBound Reproducibility Guide

This guide provides the complete, step-by-step instructions to reproduce all empirical benchmark results reported for **ReBound** on the LAV-DF dataset across all 5 evaluation splits.

---

## 1. Environment & Setup

### Prerequisites
* Linux (Ubuntu 20.04+ recommended)
* NVIDIA GPUs with CUDA 11.8+ / 12.0+
* Conda / Miniconda

### Environment Installation
```bash
# Clone the repository and switch to the dashlab branch
git clone -b dashlab git@github.com:Helixometry-DASH/AVIGIL.git ReBound
cd ReBound

# Create and activate environment
conda create -n cvpr python=3.10 -y
conda activate cvpr

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project requirements
pip install -r requirements.txt
export PYTHONPATH=src
```

---

## 2. Dataset Setup & Verification

Configure `configs/server.json` to point to your local dataset root:
```json
{
  "dataset_root": "/path/to/LAV-DF",
  "workspace": "runs/seed41",
  "device": "cuda"
}
```

Run preflight checks to verify data integrity and hardware availability:
```bash
python -m rebound preflight --config configs/server.json
```

---

## 3. Training Pipeline (From Scratch)

ReBound trains in structured stages preserving zero target leakage:

### Stage A: Unsupervised Real-Referenced Representations
Anchors multi-modal latent representations strictly to authentic speech statistics:
```bash
# Prepare manifests and split rows
python -m rebound prepare --config configs/server.json --execute

# Extract multi-modal features
python -m rebound extract --config configs/server.json --execute

# Extract genuine reference inputs
python -m rebound reference-inputs --config configs/server.json --execute

# Fit PCA projections on authentic acoustic-visual manifold
python -m rebound fit-pca --config configs/server.json --execute

# Fit Real-Referenced Multi-Scale CCA
python -m rebound fit-cca --config configs/server.json --execute

# Train genuine density estimator & normalizing transport flows
python -m rebound train-genuine --config configs/server.json --execute
python -m rebound fit-transport --config configs/server.json --execute

# Pretrain directional MMD boundary head
python -m rebound pretrain-boundary --config configs/server.json --execute
```

### Stage B: Supervised Expert Training
Trains native expert backbones with continuous sub-frame offset regression heads:
```bash
# Train Audio Expert (audio_source)
CUDA_VISIBLE_DEVICES=0 python -m rebound train-experts --config configs/server.json --source audio_source --execute

# Train Visual Expert (visual_source)
CUDA_VISIBLE_DEVICES=1 python -m rebound train-experts --config configs/server.json --source visual_source --execute

# Train Cross-Modal Invariant Alignment Heads
CUDA_VISIBLE_DEVICES=0 python -m rebound train-crossmodal --config configs/server.json --source audio_source --execute
CUDA_VISIBLE_DEVICES=1 python -m rebound train-crossmodal --config configs/server.json --source visual_source --execute

# Train Joint Mixed Expert (mixed_source, DDP multi-GPU)
torchrun --nproc_per_node=4 -m rebound train-experts --config configs/server.json --source mixed_source --execute
```

### Stage C: Genuine EVT Tail-Ranking Calibration
Calibrates proposal confidence bounds using held-out genuine clips (zero target forgery labels):
```bash
python -m rebound calibrate --config configs/server.json --source audio_source --execute
python -m rebound calibrate --config configs/server.json --source visual_source --execute
python -m rebound calibrate --config configs/server.json --source mixed_source --execute
```

---

## 4. Benchmark Evaluation Across All 5 Splits

Run full inference and compute official canonical metrics (`temporal_metrics`) across 52,924 test clips:

```bash
# Split 1: Seen Audio (audio_source -> A, 13,191 clips)
python -m rebound infer --config configs/server.json --source audio_source --target A --execute
python -m rebound evaluate --config configs/server.json --source audio_source --target A

# Split 2: Seen Visual (visual_source -> V, 13,271 clips)
python -m rebound infer --config configs/server.json --source visual_source --target V --execute
python -m rebound evaluate --config configs/server.json --source visual_source --target V

# Split 3: Unseen Audio Zero-Shot (visual_source -> A, 13,191 clips)
python -m rebound infer --config configs/server.json --source visual_source --target A --execute
python -m rebound evaluate --config configs/server.json --source visual_source --target A

# Split 4: Unseen Visual Zero-Shot (audio_source -> V, 13,271 clips)
python -m rebound infer --config configs/server.json --source audio_source --target V --execute
python -m rebound evaluate --config configs/server.json --source audio_source --target V

# Split 5: Joint Mixed (mixed_source -> MIXED, 25,930 clips)
python -m rebound infer --config configs/server.json --source mixed_source --target MIXED --execute
python -m rebound evaluate --config configs/server.json --source mixed_source --target MIXED
```

---

## 5. Expected Official Results

| Split | Target Set | Test Clips | AP@50 | AP@75 | **AP@95** | AR@100 |
|---|---|---|---|---|---|---|
| **$A \to A$ (Seen Audio)** | Audio manipulations | 13,191 | 97.49% | 94.99% | **90.21%** | 96.18% |
| **$V \to V$ (Seen Visual)** | Visual manipulations | 13,271 | 97.70% | 93.69% | **88.66%** | 94.70% |
| **$V \to A$ (Unseen Audio)** | Blind cross-modal transfer | 13,191 | **31.56%** | **17.54%** | 1.47% | **31.84%** |
| **$A \to V$ (Unseen Visual)** | Blind cross-modal transfer | 13,271 | **30.04%** | **13.12%** | 0.70% | **23.61%** |
| **$\text{MIX} \to \text{MIX}$ (Joint)**| Joint multi-modal fakes | 25,930 | 95.96% | 93.55% | **89.07%** | 95.62% |
| **$\text{MIX} \to \text{MIX}$ (Router)**| MoE Private Expert Route | 25,930 | **97.71%** | **96.07%** | **91.26%** *(92.30% peak)* | **98.33%** |

Frozen official metrics are archived in `runs/seed41/FINAL_RESULTS.json`.
