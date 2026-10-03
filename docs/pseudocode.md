# Appendix A.1 — Complete pipeline pseudocode

From the manuscript.

```
Input:  Raw document D
Output: Chunk set C = c1, c2, ..., cn

 1:   D = Preprocess(D)                              # Heuristic layout normalization
 2:   has_title = StructureDetection(D)              # Title hierarchy detection
 3:   D = NoiseRemoval(D)                            # Recurrent header and footer removal
 4:   if has_title:
 5:       B = TitleBasedSplit(D)                     # Title-anchored segmentation
 6:   else:
 7:       B = SemanticMerge(D)                        # Coherence-oriented semantic merging
 8:   for bi in B:
 9:       h2i = GenerateH2Summary(bi)                 # Local narrative summarization
10:   R = TopicFlowClustering(B, {h2i})               # Topic-flow-based region clustering
11:   for rj in R:
12:       h1j = GenerateH1Summary(rj)                 # Region-level narrative summarization
13:   C = []                                          # Initialize empty list
14:   for rj in R:
15:       C.append(CreateSummaryChunk(h1j))           # Region-summary chunk construction
16:       for bi in rj:
17:           segs = SlidingWindow(bi, Ltarget, Loverlap, Lback)
18:           for seg in segs:
19:               C.append(Concat(h2i, seg))          # Summary-augmented chunking
```

Implementation mapping:

| Pseudocode stage | Script |
|---|---|
| `Preprocess`, `StructureDetection`, `NoiseRemoval` | `src/chunking/1-merge_punctuation_marks.py`, `2-has_title.py`, `3-clear_noise.py` |
| `TitleBasedSplit` | `4.2-Split_Level.py` + `5-Outline_A.py` |
| `SemanticMerge` | `4.1-merge_line_break.py` |
| `GenerateH2Summary` (+ chained accumulation for long blocks) | `5-Outline_A.py` |
| `TopicFlowClustering` / region assignment | `6-Merge_H2.py` (two-stage merge) |
| `GenerateH1Summary` | `7-Outline_B.py` |
| `SlidingWindow` + `Concat` (both slice types) | `8-split.py` |
| Orchestration | `main.py` |

Figures: the pipeline figure (Fig. 1) shows these stages; its draw.io source is not
redistributed, the exported vector is embedded in the manuscript.
