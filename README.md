# ReBound: Real-Referenced Boundary Transfer for High-Precision Audio-Visual Deepfake Localization

ReBound transforms audio-visual temporal forgery localization from fixed-grid classification into continuous-time sub-frame regression, eliminating frame quantization errors and enabling zero-shot cross-modal transfer.

---

## 1. Quick Inference With Pretrained Weights

Pretrained weights are stored under `weights/rebound/`:
* `experts_audio.pt`: Dedicated Audio Expert (20.4 MB, 90.21% AP@95)
* `experts_visual.pt`: Dedicated Visual Expert (20.4 MB, 88.66% AP@95)
* `experts_mixed.pt`: Joint Multi-Modal Expert (20.4 MB, 89.07% Joint AP@95, 92.30% MoE Router AP@95)
* `cca.pt`: Real-Referenced Canonical Correlation Analysis reference
* `transport.pt`: Autoregressive genuine speech normalizing flows

### Single Video Inference via Python

```python
import torch
from rebound.native import NativeExperts
from rebound.assemble import blend_predictions
from rebound.decode import decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Pretrained ReBound Expert Model
model = NativeExperts("mixed_source").to(device)
ckpt = torch.load("weights/rebound/experts_mixed.pt", map_location=device)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

# 2. Forward pass with extracted features (audio, video)
# output contains continuous millisecond boundary offsets: dt * a * tanh(u_t)
with torch.no_grad():
    prediction = model(audio, video, video_valid, audio_length, video_frames)

# 3. Continuous-time proposal decoding
proposals = decode(
    prediction,
    frame_threshold=0.5,
    nms_iou=0.3,
    budget=100,
    refine='centroid'  # sub-frame continuous boundary refinement
)
# Returns list of proposals: [confidence_score, start_time_sec, end_time_sec]
print("Detected Forgery Intervals:", proposals)
```

### Full Benchmark Inference Across 5 Splits

```bash
export PYTHONPATH=src

# 1. Seen Audio Evaluation (audio_source -> A)
python -m rebound infer --config configs/server.json --source audio_source --target A --execute
python -m rebound evaluate --config configs/server.json --source audio_source --target A

# 2. Seen Visual Evaluation (visual_source -> V)
python -m rebound infer --config configs/server.json --source visual_source --target V --execute
python -m rebound evaluate --config configs/server.json --source visual_source --target V

# 3. Unseen Audio Zero-Shot Transfer (visual_source -> A)
python -m rebound infer --config configs/server.json --source visual_source --target A --execute
python -m rebound evaluate --config configs/server.json --source visual_source --target A

# 4. Unseen Visual Zero-Shot Transfer (audio_source -> V)
python -m rebound infer --config configs/server.json --source audio_source --target V --execute
python -m rebound evaluate --config configs/server.json --source audio_source --target V

# 5. Joint Multi-Modal Evaluation (mixed_source -> MIXED, 25,930 clips)
python -m rebound infer --config configs/server.json --source mixed_source --target MIXED --execute
python -m rebound evaluate --config configs/server.json --source mixed_source --target MIXED
```

---

## 2. Key Performance Benchmarks (LAV-DF Test Set: 52,924 Clips)

| Split | Target Set | Metric | Vanilla AVIGIL | ReBound (Ours) | Absolute Gain |
|---|---|---|---|---|---|
| **$A \to A$ (Seen Audio)** | 13,191 clips | AP@50<br>AP@75<br>**AP@95**<br>AR@100 | 94.90%<br>93.35%<br>80.37%<br>93.10% | **97.49%**<br>**94.99%**<br>**90.21%**<br>**96.18%** | +2.59%<br>+1.64%<br>**+9.84%**<br>+3.08% |
| **$V \to V$ (Seen Visual)** | 13,271 clips | AP@50<br>AP@75<br>**AP@95**<br>AR@100 | 97.37%<br>93.41%<br>19.40%<br>89.00% | **97.70%**<br>**93.69%**<br>**88.66%**<br>**94.70%** | +0.33%<br>+0.28%<br>**+69.26%**<br>+5.70% |
| **$V \to A$ (Unseen Audio)** | 13,191 clips | **AP@50**<br>**AP@75**<br>AP@95<br>**AR@100** | 3.80%<br>2.50%<br>0.61%<br>3.17% | **31.56%**<br>**17.54%**<br>1.47%<br>**31.84%** | **+27.76%**<br>**+15.04%**<br>+0.86%<br>**+28.67%** (10x AR) |
| **$A \to V$ (Unseen Visual)** | 13,271 clips | **AP@50**<br>**AP@75**<br>AP@95<br>**AR@100** | 2.96%<br>2.36%<br>0.21%<br>2.33% | **30.04%**<br>**13.12%**<br>0.70%<br>**23.61%** | **+27.08%**<br>**+10.76%**<br>+0.49%<br>**+21.28%** (10x AR) |
| **$\text{MIX} \to \text{MIX}$ (Joint)** | 25,930 clips | AP@50<br>AP@75<br>**AP@95**<br>AR@100 | 95.69%<br>93.27%<br>61.23%<br>92.22% | **95.96%** (Joint)<br>**93.55%** (Joint)<br>**89.07%** (Joint)<br>**95.62%** (Joint)<br>*(Router: 92.30% AP95)* | +0.27%<br>+0.28%<br>**+27.84%**<br>+3.40%<br>*(Router: +31.07% AP95)* |

---

## 3. Project Documentation Links

* [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): Complete pipeline instructions to train from scratch and reproduce all metrics.
* [`AVIGIL_VS_REBOUND.md`](AVIGIL_VS_REBOUND.md): Side-by-side comparison, ablation studies, and architectural differences vs. AVIGIL.
* [`ARCHITECTURE.md`](ARCHITECTURE.md): Mathematical formulations, LaTeX proofs, and architecture schemas.
* [`PRESENTATION_SLIDES_IJCAI.md`](PRESENTATION_SLIDES_IJCAI.md): Presentation slide deck for research meetings and advisor updates.
