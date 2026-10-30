# Presentation: From AVIGIL to ReBound
## High-Precision Temporal Audio-Visual Deepfake Localization
**Target Venue:** IJCAI 2026 | **Advisor Update**

---

## Slide 1: Title & Executive Overview

### Project Context
* **Baseline Foundation:** AVIGIL (developed by co-author Girish).
* **Our Proposed Method:** **ReBound** (Continuous Sub-Frame Localization & Invariant Transfer).
* **Target Conference:** **IJCAI** (re-targeted following server NAS downtime during CVPR deadline).
* **Current Milestone:** Completed 52,924-clip full benchmark on LAV-DF across all 5 evaluation splits.

---

## Slide 2: Why AVIGIL Struggles at High Precision (The Problems)

```
[Problem 1: The 40ms Video Quantization Wall]
- AVIGIL detects forgery on discrete 25 fps video frames (40ms steps).
- At strict IoU >= 0.95, a 1-frame (40ms) error drops score to 0.
- Result: AVIGIL AP@95 collapses to 19.4% on visual fakes.

[Problem 2: Zero-Shot Cross-Modal Collapse]
- AVIGIL's cross-attention trains directly on source artifacts (e.g. lip jitter).
- When tested on unseen audio splices, features fail completely: AP@50 = 3.8%.

[Problem 3: Modality Cross-Talk Interference]
- When both audio AND video are fake, cross-attention confuses signals.
- Dual-fake classification accuracy drops from 99.6% down to 78.1%.
```

---

## Slide 3: What ReBound Adds (Our Core Innovations)

```
                       ┌───────────────────────────────────────────────┐
                       │          ReBound Core Contributions           │
                       └───────────────────────────────────────────────┘
                                              │
         ┌───────────────────┬────────────────┴───────────────────┬───────────────────┐
         ▼                   ▼                                    ▼                   ▼
[Sub-Frame Regression] [Real-Ref CCA]                      [Directional MMD]   [Private MoE Router]
b = t + dt*a*tanh(u)   Invariance to forgery flaws         Kernels (80-320ms)  Separate paths for
Continuous ms bounds   Anchored to authentic speech        Isolate boundary    audio, video & joint
Restores AP95 to 91%+  8x unseen transfer gain             transition dynamics Eliminates interference
```

1. **Continuous Sub-Frame Neural Head:** Regresses exact millisecond boundary offsets $\Delta_t a \tanh(u_t)$, breaking the 40ms quantization barrier.
2. **Real-Referenced CCA Bottleneck:** Aligns multimodal representations strictly to genuine speech statistics; detects desynchronization without memorizing specific forgery generator flaws.
3. **Multi-Scale Directional MMD:** Temporal kernel profiles (80ms, 160ms, 320ms) explicitly detect left/right boundary transition contrast.
4. **MoE Private Expert Routing:** Dispatches unimodal signals to dedicated private experts and joint signals to shared CCA, removing cross-talk.

---

## Slide 4: Current Experimental Results on LAV-DF (52,924 Clips)

### Full 5-Split Benchmark (Canonical Metrics)

| Evaluation Split | Challenge Type | Test Clips | AVIGIL Baseline | ReBound (Ours) | Absolute Gain |
|---|---|---|---|---|---|
| **$A \to A$ (Seen Audio)** | Boundary Sharpness | 13,191 | AP95: 80.37% | **AP95: 90.21%** | **+9.84%** |
| **$V \to V$ (Seen Visual)** | 40ms Quantization | 13,271 | AP95: 19.40% | **AP95: 88.66%** | **+69.26%** |
| **$V \to A$ (Unseen Audio)** | Zero-Shot Transfer | 13,191 | AP50: 3.80% | **AP50: 31.56%** | **+27.76% (8x gain)** |
| **$A \to V$ (Unseen Visual)** | Zero-Shot Transfer | 13,271 | AP50: 2.96% | **AP50: 30.04%** | **+27.08% (10x gain)** |
| **$\text{MIX} \to \text{MIX}$ (Joint)**| Modality Cross-Talk| 25,930 | AP95: 61.23%<br>Union95: 49.72% | **AP95: 91.26%**<br>**Union95: 91.26%** | **+30.03%**<br>**+41.54%** |

*Modality Classification on Dual Fakes: AVIGIL = 78.07% $\to$ **ReBound = 100.00%**.*

---

## Slide 5: Efficiency & Parameter Footprint

* **Inference Speed & FLOPs:** Exact 1-to-1 parity with AVIGIL (5.32M active parameters per clip).
* **Additional Parameters in Head:** Offset heads add only **1,280 parameters** (<0.002M).
* **Architecture Verification:** Validated on single shared model (ReBound Joint: 89.07% AP@95) and multi-expert routing (ReBound Router: 91.26% – 92.30% AP@95).

---

## Slide 6: Dataset Expansion Plan (Roadmap for Full Paper)

To ensure high-impact acceptance at **IJCAI**, we are expanding evaluation across 6 diverse datasets:

```
[Phase 1: Completed]
- LAV-DF: Full 5-split benchmark finished (52,924 clips).

[Phase 2: Multimodal Benchmarks]
- AV-1M: Large-scale audio-visual dataset testing scalability.
- ArEnAV: Cross-lingual Arabic-English AV dataset testing language invariance.
- CHAV-DF (Springer 2025): Latest multi-generator Chinese-English benchmark.
- FakeAVCeleb: Diverse celebrity deepfakes across 4 distinct synthesis engines.

[Phase 3: Unimodal Stress Testing]
- FaceForensics++ (FF++): High-resolution video face manipulation testing.
- ASVspoof: Audio spoofing and synthetic speech boundary testing.
```

---

## Slide 7: Next Steps & IJCAI Submission Timeline

1. **Feature Extraction & Ingestion:**
   * Download and extract AV-1M, ArEnAV, CHAV-DF, FF++, and ASVspoof features.
2. **Zero-Shot Cross-Dataset Validation:**
   * Evaluate LAV-DF-trained ReBound directly on CHAV-DF and FakeAVCeleb without retraining to prove cross-dataset generalization.
3. **Paper Drafting:**
   * Consolidate theory (CCA invariance bound proof + continuous boundary localization).
   * Format paper according to IJCAI submission guidelines.
