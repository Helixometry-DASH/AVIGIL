# ReBound seed 41 results

Completed 2026-10-28 01:59 KST. All four training/inference directions completed. Requested accuracy targets were not met.

Combined model; canonical GT-first, strict-IoU AP, percentages:

| Direction | AP50 | AP75 | AP95 |
|---|---:|---:|---:|
| A → A (seen) | 90.08 | 87.45 | 75.46 |
| A → V (unseen) | 2.02 | 1.17 | 0.11 |
| V → A (unseen) | 10.26 | 4.23 | 0.88 |
| V → V (seen) | 95.32 | 88.67 | 73.85 |

Private forensic route alone:

| Direction | AP75 | AP95 |
|---|---:|---:|
| A → A (seen) | 93.35 | 82.73 |
| V → V (seen) | 90.55 | 78.17 |

The shared boundary routes have very low AP75, and combining them reduces seen-modality AP. This run does not validate the proposed unseen-modality transfer hypothesis. The exact cause requires further diagnostics; these results alone do not establish that the general idea is impossible.

Targets: seen AP75 ≥95%, seen AP95 ≥86%; unseen AP75 ≥30–40% in both directions. These are one-seed results, not a multi-seed confirmation.

Full metrics and checkpoint signatures: `FINAL_RESULTS.json`. Server originals: `/home/inzi15/ReBound/runs/seed41/FINAL_RESULTS.json`.
