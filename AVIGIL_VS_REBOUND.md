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
