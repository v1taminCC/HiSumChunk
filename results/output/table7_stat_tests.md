# Table 7 (= Appendix A.2). Composite-score difference vs HiSumChunk

Per-question paired differences, works resampled as clusters (20,000 resamples, seed 2026), Holm correction over the six comparisons. Rows sorted by raw p.

| Comparison           | Delta (pts) | 95% CI           | p (raw) | p (Holm) | Verdict |
|----------------------|-------------|------------------|---------|----------|---------|
| Fixed-1024W          | -5.416      | [-7.847, -3.096] | 0.0000  | 0.0003   | ***     |
| LLMSemantic          | -4.001      | [-6.113, -2.051] | 0.0001  | 0.0005   | ***     |
| SemanticEmbedding    | -4.795      | [-7.631, -1.970] | 0.0004  | 0.0018   | **      |
| LumberChunker        | -4.145      | [-7.344, -0.511] | 0.0131  | 0.0394   | *       |
| RAPTOR               | -1.693      | [-3.797, +0.540] | 0.0609  | 0.1219   | n.s.    |
| Contextual Retrieval | -1.906      | [-4.502, +1.614] | 0.1246  | 0.1246   | n.s.    |

Self-check of Holm p-values against the paper:

- vs Fixed-1024W: paper 0.0003 vs reproduced 0.0003 -> OK
- vs LumberChunker: paper 0.0394 vs reproduced 0.0394 -> OK
- vs LLMSemantic: paper 0.0005 vs reproduced 0.0005 -> OK
- vs SemanticEmbedding: paper 0.0018 vs reproduced 0.0018 -> OK
- vs RAPTOR: paper 0.1219 vs reproduced 0.1219 -> OK
- vs Contextual Retrieval: paper 0.1246 vs reproduced 0.1246 -> OK