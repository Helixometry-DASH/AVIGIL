# ReBound vs. AVIGIL: Technical Additions, Ablations, and Official Benchmark Results

## Executive Summary
ReBound transforms the AVIGIL baseline into a continuous-time, zero-leakage temporal forgery localization framework. Evaluated on the identical **52,924 test clips** across all 5 benchmark splits with canonical metrics (`temporal_metrics`), ReBound eliminates the 40ms frame quantization wall (+69.26% AP@95 on visual deepfakes) and establishes robust zero-shot cross-modal transfer (+27.76% AP@50, +15.04% AP@75) where pure AVIGIL fails (~3% AP).

---

## 1. Why & How AVIGIL Becomes ReBound

```
[AVIGIL Baseline]
Fixed 40ms Temporal Grid ──> Discrete Frame Argmax ──> Global Heuristic Thresholds ──> Brittle Boundaries (19.4% AP95)
Unconstrained Cross-Attn ──> Memorizes Forgery Flaws ──> Zero-Shot Cross-Modal Transfer Collapses (2-3% AP)

                              ↓↓↓ [Architectural Redesign] ↓↓↓

[ReBound]
1. Continuous Sub-Frame Head: Regresses exact continuous boundary offsets dt.
2. Real-Referenced CCA Bottleneck: Dual streams anchored strictly to authentic speech manifold.
3. Multi-Scale Directional MMD: Temporal kernels (80ms/160ms/320ms) isolating directional boundary transitions.
4. Genuine-Calibrated Tail-Ranking: Extreme-value tail normalization against genuine speech variance.
```

### Module Evolution Breakdown:
1. **Discrete Grid -> Continuous Sub-Frame Regression:**
   - *AVIGIL:* Frame-level argmax classifies discrete windows (25 fps = 40 ms steps). At IoU >= 0.95 on short 0.5s segments, a single 40 ms frame truncation drops IoU < 0.95, collapsing AP@95 to **19.40%**.
   - *ReBound:* Adds continuous offset regression dt * a * tanh(u_t), adjusting discrete boundary edges by continuous sub-frame shifts. Restores AP@95 to **88.66%** (+69.26% absolute gain).
2. **Dense Cross-Attention -> Real-Referenced CCA:**
   - *AVIGIL:* Bidirectional cross-attention trains directly on source forgery artifacts (e.g. video face warping artifacts). When tested on unseen audio splices, features are completely uninformative (AP50: **3.80%**, AR100: **3.17%**).
   - *ReBound:* Dual audio-visual streams project into shared canonical space anchored only on authentic speech correlations. Measures desynchronization divergence rather than generator artifacts, delivering **31.56% AP@50** and **17.54% AP@75** without seeing a single target forgery label.
3. **Global Threshold -> Genuine-Calibrated Tail Ranking:**
   - *AVIGIL:* Global raw confidence values fail on expressive speech, triggering false positives.
   - *ReBound:* Fits genuine reference empirical distributions, ranking proposals by deviation from genuine tail bounds.

---

## 2. What Gains & What Doesn't Gain

### What Gains:
- **Seen Visual Boundary Precision (AP@95):** Massive **+69.26%** gain (19.40% -> 88.66%). Continuous sub-frame regression completely overcomes frame quantization.
- **Seen Audio Boundary Precision (AP@95):** Solid **+9.84%** gain (80.37% -> 90.21%).
- **Seen Joint Mixed Detection (AP@95):** **+27.84%** gain (61.23% -> 89.07%), and **+31.07%** gain (92.30%) with private expert routing.
- **Unseen Cross-Modal Transfer (AP@50 & AP@75):**
  - V -> A: **+27.76% AP@50** (3.80% -> 31.56%), **+15.04% AP@75** (2.50% -> 17.54%), **+28.67% AR@100** (3.17% -> 31.84%).
  - A -> V: **+27.08% AP@50** (2.96% -> 30.04%), **+10.76% AP@75** (2.36% -> 13.12%), **+21.28% AR@100** (2.33% -> 23.61%).

### What Doesn't Gain (Inherent Physical Limits):
- **Unseen Ultra-Strict AP@95 (Capped at ~1.5%):**
  - While coarse localization (AP@50/75) jumps 7x-10x, AP@95 on unseen cross-modal transfer remains low (AVIGIL: 0.21-0.61%, ReBound: 0.70-1.47%).
  - *Root Cause:* Visual deepfakes (e.g. face swaps) and audio deepfakes (e.g. voice clones, acoustic splices) share zero physical micro-boundary cues. A model trained only on acoustic splices cannot guess the exact millisecond video blending frame without having seen video manipulations during training.
- **Seen AP@50 Saturation:**
  - Both AVIGIL and ReBound already score ~95-97% at IoU 0.50 on seen modalities. Rough segment detection was already solved; ReBound solves boundary sharpness and zero-shot transfer.

---

## 3. Official 5-Split Benchmark Results (Side-by-Side)

Evaluated across **52,924 test clips** with official canonical evaluation script (`temporal_metrics`).

| Split | Target Set | Metric | Vanilla AVIGIL | ReBound (Ours) | Absolute Gain |
|---|---|---|---|---|---|
| **A -> A (Seen Audio)** | 13,191 clips | AP@50<br>AP@75<br>**AP@95**<br>AR@100 | 94.90%<br>93.35%<br>80.37%<br>93.10% | **97.49%**<br>**94.99%**<br>**90.21%**<br>**96.18%** | +2.59%<br>+1.64%<br>**+9.84%**<br>+3.08% |
| **V -> V (Seen Visual)** | 13,271 clips | AP@50<br>AP@75<br>**AP@95**<br>AR@100 | 97.37%<br>93.41%<br>19.40%<br>89.00% | **97.70%**<br>**93.69%**<br>**88.66%**<br>**94.70%** | +0.33%<br>+0.28%<br>**+69.26%**<br>+5.70% |
| **V -> A (Unseen Audio)** | 13,191 clips | **AP@50**<br>**AP@75**<br>AP@95<br>**AR@100** | 3.80%<br>2.50%<br>0.61%<br>3.17% | **31.56%**<br>**17.54%**<br>1.47%<br>**31.84%** | **+27.76%**<br>**+15.04%**<br>+0.86%<br>**+28.67%** (10x AR) |
| **A -> V (Unseen Visual)** | 13,271 clips | **AP@50**<br>**AP@75**<br>AP@95<br>**AR@100** | 2.96%<br>2.36%<br>0.21%<br>2.33% | **30.04%**<br>**13.12%**<br>0.70%<br>**23.61%** | **+27.08%**<br>**+10.76%**<br>+0.49%<br>**+21.28%** (10x AR) |
| **MIX -> MIX (Joint)** | 25,930 clips | AP@50<br>AP@75<br>**AP@95**<br>AR@100 | 95.69%<br>93.27%<br>61.23%<br>92.22% | **95.96%** (Joint)<br>**93.55%** (Joint)<br>**89.07%** (Joint)<br>**95.62%** (Joint)<br>*(Router: 92.30% AP95)* | +0.27%<br>+0.28%<br>**+27.84%**<br>+3.40%<br>*(Router: +31.07% AP95)* |

---

## 4. Comprehensive Mixed $	o$ Mixed Benchmark: Author Release vs. Re-evaluation vs. ReBound

Comparison against the author's official released checkpoint (`weights/avigil_m5/model.pt`, report from `test_metrics.json`), our full 25k-clip re-evaluation of AVIGIL, and ReBound on identical test data:

| Metric | AVIGIL Official (`model.pt`) | AVIGIL Re-eval (Our Eval, Argmax) | ReBound (Joint) | ReBound (Router) | Net Gain (Router vs. AVIGIL) |
|---|---|---|---|---|---|
| **AP@50** | 94.23% | 95.69% | 95.96% | **97.71%** | **+2.02% ~ +3.48%** |
| **AP@75** | 91.53% | 93.27% | 93.55% | **96.07%** | **+2.80% ~ +4.54%** |
| **AP@90** | 88.64% | 79.20% | 91.33% | **93.86%** | **+5.22% ~ +14.66%** |
| **AP@95** | 85.60% | **61.23%** *(quantized)* | 89.07% | **91.26%** *(92.30%)* | **+5.66% ~ +30.03%** |
| **AR@10** | 93.17% | 91.80% | 96.05% | **98.00%** | **+4.83% ~ +6.20%** |
| **Union AP@50** | 93.37% | 95.69% | 95.96% | **97.71%** | **+2.02% ~ +4.34%** |
| **Union AP@75** | 90.18% | 93.27% | 93.55% | **96.07%** | **+2.80% ~ +5.89%** |
| **Union AP@95** | **49.72%** *(collapse)* | **61.23%** | 89.07% | **91.26%** | **+30.03% ~ +41.54%** |
| **Modality Acc: Visual** | 99.78% | 98.40% | 100.00% | **100.00%** | +0.22% ~ +1.60% |
| **Modality Acc: Audio** | 99.54% | 97.80% | 100.00% | **100.00%** | +0.46% ~ +2.20% |
| **Modality Acc: Both** | **78.07%** *(interference)* | **76.50%** | 100.00% | **100.00%** | **+21.93% ~ +23.50%** |

### Key Observations:
1. **AVIGIL Union AP@95 Collapse (49.72%):** When evaluating multi-modal unions across the recording timeline, AVIGIL's cross-attention misaligns boundaries, dropping AP@95 by 35.88% relative to per-modality scoring. ReBound's continuous sub-frame regression maintains 91.26% (+41.54% absolute improvement).
2. **Dual-Manipulation Interference (78.07%):** In clips where both audio and video are tampered with, AVIGIL's modality accuracy collapses by over 21% due to bidirectional attention cross-talk. ReBound's Private Expert Router processes unimodal signals through dedicated expert pathways, eliminating cross-talk and achieving 100.0% modality discrimination.
3. **Quantization Wall (61.23%):** Evaluating AVIGIL with standard discrete frame argmax incurs severe frame truncation penalties at strict IoU thresholds, dropping AP@95 to 61.23%.

---

## 5. Comprehensive Ablation Studies

### A. ReBound Component Deductive Ablations
Isolating the quantitative impact of each ReBound novelty on test benchmarks:

| Exp ID | Ablation Description | Test Split | AP@50 | AP@75 | **AP@95** | AR@100 | Finding & Impact |
|---|---|---|---|---|---|---|---|
| **RB-0** | **Full ReBound** (Continuous + CCA + MMD + Calib) | $V 	o V$ (Seen)<br>$V 	o A$ (Unseen) | 97.70%<br>31.56% | 93.69%<br>17.54% | **88.66%**<br>1.47% | 94.70%<br>31.84% | Full model performance baseline |
| **RB-1** | **w/o Continuous Sub-Frame Head** (Discrete 40ms Grid) | $V 	o V$ (Seen)<br>$A 	o A$ (Seen) | 97.77%<br>97.52% | 93.58%<br>94.98% | **25.79%**<br>89.79% | 87.08%<br>96.14% | **-62.87% AP@95 drop** on 40ms video grid; proves sub-frame head resolves quantization barrier. |
| **RB-2** | **w/o Real-Referenced CCA** (Dense Cross-Attention) | $V 	o A$ (Unseen)<br>$A 	o V$ (Unseen) | 1.19%<br>0.10% | **0.45%**<br>**0.01%** | 0.00%<br>0.00% | 4.08%<br>0.50% | **-17.09% AP@75 collapse**; proves dense cross-attention memorizes source artifacts and has near-zero transfer. |
| **RB-Priv** | **Private Route Only** (w/o Shared Invariant Route) | $V 	o V$ (Seen)<br>$V 	o A$ (Unseen) | 97.61%<br>3.84% | 93.65%<br>2.52% | 88.70%<br>0.84% | 94.60%<br>3.26% | Unimodal route alone fails completely on unseen (3.84% AP@50), matching AVIGIL (3.80%). |
| **RB-Shared**| **Shared CCA Route Only** (w/o Private Route) | $V 	o A$ (Unseen) | 29.73% | 16.04% | 0.47% | 30.12% | Supplies **94% of total unseen cross-modal transfer** signal. |

---

### B. AVIGIL Core Architectural & Grid Ablations

| Exp ID | AVIGIL Configuration | Grid Alignment | AP@50 | AP@75 | **AP@95** | AR@100 | Finding & Impact |
|---|---|---|---|---|---|---|---|
| **AV-M5** | **Base Released AVIGIL** (Bi-Cross-Attn + Gating) | Audio (15.2 ms)<br>Visual (40.0 ms) | 94.90% (A)<br>97.37% (V) | 93.35% (A)<br>93.41% (V) | 80.37% (A)<br>**19.40%** (V) | 93.10% (A)<br>89.00% (V) | Video branch suffers massive 19.4% AP@95 failure due to 40ms grid. |
| **AV-G0** | **40ms Video Grid Downsampling** | Video (40.0 ms) | 94.89% (A) | 93.27% (A) | **67.52%** (A) | 92.31% (A) | **-12.85% AP@95 drop** when audio is forced to 40ms grid; confirms grid size dictates discrete AP@95 ceiling. |
| **AV-Calib**| **AVIGIL + Genuine EVT Calibration** | Native Grids | 94.90% (A)<br>97.37% (V) | 93.35% (A)<br>93.41% (V) | 80.37% (A)<br>19.40% (V) | 93.10% (A)<br>89.00% (V) | EVT tail normalization preserves top-rank discrimination but cannot recover missing sub-frame boundary alignment. |
