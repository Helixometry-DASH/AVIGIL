# ReBound vs. AVIGIL: Technical Additions and Accuracy Breakthroughs

## Executive Summary
ReBound extends the AVIGIL baseline into a high-precision, zero-leakage temporal forgery localization framework. While baseline AVIGIL struggled at strict IoU thresholds (AP@95), ReBound achieves **90.21% AP@95** on audio and **88.66% AP@95** on visual manipulations, while enabling zero-shot unseen cross-modal transfer.

---

## 1. What We Add Over AVIGIL (Core Novelties)

* **Continuous Sub-Frame Boundary Heads:**
  * *AVIGIL:* Frame-level discrete classification on fixed temporal grids (~40ms/80ms steps).
  * *ReBound:* Continuous offset regression $\widehat{b}_t = t_t + \Delta_t a \tanh(u_t)$ predicting exact continuous boundary timestamps. Eliminates grid quantization error.
* **Real-Referenced Invariant Bottleneck (Multi-Scale CCA & Flows):**
  * *AVIGIL:* Unconstrained multimodal fusion that overfits to source-specific manipulation artifacts (causing 0% transfer on unseen modalities).
  * *ReBound:* Dual-stream representations anchored strictly to genuine speech statistics (`Real` fitting families). Measures divergence from authentic speech distributions rather than memorizing generator flaws.
* **Directional Kernel Profiles (Multi-Scale MMD):**
  * *AVIGIL:* Pure feed-forward classification without explicit boundary dynamics.
  * *ReBound:* Multi-scale temporal Maximum Mean Discrepancy (MMD) profiles across 80ms, 160ms, and 320ms windows that mathematically isolate left-to-right and right-to-left boundary transitions.
* **Genuine-Calibrated Tail-Ranking:**
  * *AVIGIL:* Global heuristic confidence thresholds prone to high false-alarm rates on expressive genuine speech.
  * *ReBound:* Extreme-value empirical tail-ranking calibrated on held-out genuine reference clips, normalizing proposal confidence against authentic speech variance.
* **Strict Zero-Target-Leakage Protocol:**
  * Enforces source isolation where models evaluated on unseen modalities never encounter target manipulation labels during fitting, training, or calibration.

---

## 2. Why Accuracy Increases So Dramatically

### A. Seen Modality AP@95 Jump (from ~50% to ~90%)
1. **Elimination of Quantization Error:**
   * At IoU 0.95 on short intervals (e.g., 0.5s – 1.0s), an error of just 1–2 video frames (40–80 ms) drops the IoU below 0.95, completely zeroing out the AP@95 score.
   * ReBound's continuous offset regression aligns predictions to exact millisecond boundaries, converting near-misses into high-precision IoU $\ge 0.95$ detections.
2. **Focal Boundary Localization Loss:**
   * ReBound penalizes boundary uncertainty with combined Dice, focal boundary terms, and Smooth-L1 offsets, preventing boundary probability plateaus.

### B. Unseen Modality Transfer (from 0.45% to 17.54% AP@75)
1. **Breaking the Cross-Attention Overfitting Trap:**
   * Naive cross-attention achieves high validation scores by memorizing source cues (e.g., lip jitter), which collapse to **0.45%** on unseen audio splices.
   * ReBound's symmetric Real-Referenced CCA extracts invariant acoustic-visual desynchronization signals, sustaining **17.54% AP@75** across 13k+ blind test clips without a single target label.

---

## 3. Comparative Performance Matrix

| Metric | AVIGIL Baseline | ReBound (Ours) | Absolute Gain |
|---|---|---|---|
| **Seen Audio AP@95** | ~52.0% | **90.21%** | **+38.2%** |
| **Seen Visual AP@95** | ~48.5% | **88.66%** | **+40.1%** |
| **Seen Audio AP@75** | ~74.0% | **94.99%** | **+21.0%** |
| **Seen Visual AP@75** | ~71.0% | **93.69%** | **+22.7%** |
| **Unseen Transfer AP@75** | ~0.5% (dense overfit) | **17.54%** | **+17.0%** |
| **Test Scale Evaluated** | Subsets | **52,924 clips** | Full benchmark |
