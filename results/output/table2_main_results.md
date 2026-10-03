# Table 2. Main results of the seven methods

Dimensions are 1-5 LLM-as-Judge means; composite is the 100-point equal-weight average of the three dimensions. 4,038 valid answer records (RAPTOR has 576).

| Method               | Grounding | Correctness | Hall. absence | Composite | Ctx ROUGE-L | Ctx BERTScore | Ent. prec. | Ent. recall | Unsup. ent. rate |
|----------------------|-----------|-------------|---------------|-----------|-------------|---------------|------------|-------------|------------------|
| HiSumChunk (ours)    | 4.77      | 4.79        | 4.76          | 94.37     | 0.2516      | 0.8790        | 0.5462     | 0.6743      | 0.1809           |
| RAPTOR               | 4.69      | 4.72        | 4.71          | 92.67     | 0.2203      | 0.8638        | 0.4960     | 0.6691      | 0.1534           |
| Contextual Retrieval | 4.67      | 4.73        | 4.69          | 92.46     | 0.2438      | 0.8776        | 0.5388     | 0.6744      | 0.1678           |
| LumberChunker        | 4.65      | 4.65        | 4.53          | 90.22     | 0.2068      | 0.8594        | 0.5027     | 0.6898      | 0.2545           |
| LLMSemantic          | 4.63      | 4.63        | 4.58          | 90.37     | 0.2116      | 0.8601        | 0.5159     | 0.6891      | 0.2420           |
| SemanticEmbedding    | 4.60      | 4.60        | 4.55          | 89.57     | 0.2120      | 0.8603        | 0.5142     | 0.6771      | 0.2298           |
| Fixed-1024W          | 4.58      | 4.59        | 4.51          | 88.95     | 0.2144      | 0.8609        | 0.5302     | 0.6909      | 0.2018           |

Self-check vs paper: 7/7 methods match within 0.01 (two-stage rounding).