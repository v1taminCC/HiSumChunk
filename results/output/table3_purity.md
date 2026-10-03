# Table 3. Context main-thread purity (577 questions)

Purity = share of top-5 words falling in the largest chapter/scene-level unit; the largest-block column is the mean word count of that unit.

| Method               | Mean purity | Median purity | Mean top-5 words | Mean largest block | Chapters in top-5 |
|----------------------|-------------|---------------|------------------|--------------------|-------------------|
| HiSumChunk (ours)    | 58.4%       | 56.0%         | 3,306            | 770                | 3.02              |
| Contextual Retrieval | 52.5%       | 46.0%         | 2,912            | 673                | 3.31              |
| LumberChunker        | 38.0%       | 35.0%         | 8,263            | 2,776              | 4.42              |
| RAPTOR               | 38.0%       | 29.3%         | 3,542            | 1,024              | 4.29              |
| SemanticEmbedding    | 36.9%       | 34.1%         | 5,684            | 1,621              | 4.37              |
| LLMSemantic          | 36.2%       | 33.6%         | 6,725            | 1,926              | 4.40              |
| Fixed-1024W          | 34.2%       | 39.8%         | 5,048            | 1,021              | 4.22              |

Self-check vs paper:
- HiSumChunk (ours): paper mean 58.4% / words 3,306 vs reproduced 58.4% / 3,306 -> OK
- Contextual Retrieval: paper mean 52.5% / words 2,912 vs reproduced 52.5% / 2,912 -> OK
- LumberChunker: paper mean 38.0% / words 8,263 vs reproduced 38.0% / 8,263 -> OK
- RAPTOR: paper mean 38.0% / words 3,542 vs reproduced 38.0% / 3,542 -> OK
- SemanticEmbedding: paper mean 36.9% / words 5,684 vs reproduced 36.9% / 5,684 -> OK
- LLMSemantic: paper mean 36.2% / words 6,725 vs reproduced 36.2% / 6,725 -> OK
- Fixed-1024W: paper mean 34.2% / words 5,048 vs reproduced 34.2% / 5,048 -> OK

## Purity difference vs HiSumChunk (percentage points)

Book-clustered paired bootstrap, 20,000 resamples, seed 2026.

| Comparison           | Delta (pp) | 95% CI         | p     |
|----------------------|------------|----------------|-------|
| Fixed-1024W          | +24.3      | [+19.3, +28.9] | 0.000 |
| LumberChunker        | +20.4      | [+15.8, +25.3] | 0.000 |
| RAPTOR               | +20.5      | [+12.0, +28.4] | 0.000 |
| SemanticEmbedding    | +21.5      | [+15.8, +27.1] | 0.000 |
| LLMSemantic          | +22.3      | [+17.7, +26.9] | 0.000 |
| Contextual Retrieval | +5.9       | [-4.7, +14.4]  | 0.281 |

Self-check vs paper Section 6.2:
- vs Fixed-1024W: paper +24.3 [+19.3, +28.9] p=0.00 vs reproduced +24.3 [+19.3, +28.9] p=0.00 -> OK
- vs LumberChunker: paper +20.4 [+15.8, +25.2] p=0.00 vs reproduced +20.4 [+15.8, +25.3] p=0.00 -> OK
- vs RAPTOR: paper +20.5 [+12.2, +28.5] p=0.00 vs reproduced +20.5 [+12.0, +28.4] p=0.00 -> OK
- vs SemanticEmbedding: paper +21.5 [+15.8, +27.0] p=0.00 vs reproduced +21.5 [+15.8, +27.1] p=0.00 -> OK
- vs LLMSemantic: paper +22.3 [+17.7, +26.8] p=0.00 vs reproduced +22.3 [+17.7, +26.9] p=0.00 -> OK
- vs Contextual Retrieval: paper +5.9 [-4.7, +14.3] p=0.27 vs reproduced +5.9 [-4.7, +14.4] p=0.28 -> OK