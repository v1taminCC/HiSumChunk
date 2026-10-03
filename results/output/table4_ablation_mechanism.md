# Table 4. Ablation and mechanism experiments (1-5 scale)

## Panel 1: end-to-end ablation (index and retrieval rebuilt per variant)

| Variant                              | Grounding | Correctness | Hall. absence |       |
|--------------------------------------|-----------|-------------|---------------|-------|
| HiSumChunk (full)                    | 4.77      | 4.79        | 4.76          | n=577 |
| w/o H1 (no region-summary blocks)    | 4.74      | 4.75        | 4.73          | n=577 |
| w/o H2 (no local-summary prefixes)   | 4.70      | 4.73        | 4.71          | n=575 |
| w/o H1+H2 (both levels removed)      | 4.65      | 4.69        | 4.62          | n=573 |
| noise_50 (entity noise in summaries) | 4.75      | 4.78        | 4.75          | n=574 |

Self-check vs paper:
- HiSumChunk (full): paper (4.77, 4.79, 4.76) vs reproduced (4.77, 4.79, 4.76) -> OK
- w/o H1 (no region-summary blocks): paper (4.74, 4.75, 4.73) vs reproduced (4.74, 4.75, 4.73) -> OK
- w/o H2 (no local-summary prefixes): paper (4.7, 4.73, 4.71) vs reproduced (4.70, 4.73, 4.71) -> OK
- w/o H1+H2 (both levels removed): paper (4.65, 4.69, 4.62) vs reproduced (4.65, 4.69, 4.62) -> OK
- noise_50 (entity noise in summaries): paper (4.75, 4.78, 4.75) vs reproduced (4.75, 4.78, 4.75) -> OK

## Panel 2: fixed-retrieval-set mechanism arms (common valid n=575)

| Arm | Grounding | Correctness | Hall. absence | Composite |
|-----|-----------|-------------|---------------|-----------|
| A   | 4.765     | 4.793       | 4.763         | 94.35     |
| B   | 4.725     | 4.720       | 4.704         | 92.91     |
| C   | 4.677     | 4.697       | 4.683         | 92.14     |
| D   | 4.650     | 4.682       | 4.640         | 91.43     |
| P   | 4.697     | 4.710       | 4.689         | 92.46     |

A = full; B = remove H2 prefixes (H1 kept); C = remove H1 slices (back-filled); D = remove both; P = placebo (length-matched irrelevant summaries).

Self-check vs paper:
- Arm A: paper (4.765, 4.793, 4.763, 94.35) vs reproduced (4.765, 4.793, 4.763, 94.35) -> OK
- Arm B: paper (4.725, 4.72, 4.704, 92.91) vs reproduced (4.725, 4.720, 4.704, 92.91) -> OK
- Arm C: paper (4.677, 4.697, 4.683, 92.14) vs reproduced (4.677, 4.697, 4.683, 92.14) -> OK
- Arm D: paper (4.65, 4.682, 4.64, 91.43) vs reproduced (4.650, 4.682, 4.640, 91.43) -> OK
- Arm P: paper (4.697, 4.71, 4.689, 92.46) vs reproduced (4.697, 4.710, 4.689, 92.46) -> OK

## Arm deltas vs A (from data/scores/five_arm_summary.json)

| Arm | Dimension             | Delta  | 95% CI           | p (raw) | p (Holm) |
|-----|-----------------------|--------|------------------|---------|----------|
| B   | correctness           | -0.073 | [-0.120, -0.027] | 0.004   | 0.008    |
| C   | correctness           | -0.096 | [-0.152, -0.033] | 0.003   | 0.008    |
| D   | correctness           | -0.111 | [-0.162, -0.061] | 0.000   | 0.000    |
| P   | correctness           | -0.084 | [-0.143, +0.002] | 0.059   | 0.059    |
| B   | grounding             | -0.040 | [-0.099, +0.018] | 0.187   | 0.187    |
| C   | grounding             | -0.089 | [-0.160, -0.025] | 0.005   | 0.015    |
| D   | grounding             | -0.115 | [-0.167, -0.058] | 0.001   | 0.004    |
| P   | grounding             | -0.068 | [-0.119, +0.004] | 0.064   | 0.128    |
| B   | hallucination_absence | -0.059 | [-0.185, +0.100] | 0.401   | 0.834    |
| C   | hallucination_absence | -0.080 | [-0.210, +0.081] | 0.278   | 0.834    |
| D   | hallucination_absence | -0.123 | [-0.229, +0.005] | 0.060   | 0.240    |
| P   | hallucination_absence | -0.075 | [-0.201, +0.112] | 0.343   | 0.834    |

Directional stability: arm D shows the largest drop on every dimension; arm P falls below arm B on the composite (92.46 < 92.91), which supports the report's directional conclusion that the benefit comes from summary-content correspondence.