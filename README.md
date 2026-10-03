# HiSumChunk

Code and resources for the paper **"HiSumChunk: Hierarchical Summary-Guided Semantic
Chunking for Context-Preserving Long-Document RAG"**.

HiSumChunk targets a failure mode of long-narrative retrieval-augmented generation:
after chunking, retrieval units lose the description of their semantic level, so
abstract questions fail to match. Instead of changing *where* to cut, HiSumChunk
generates two levels of summaries bottom-up (an H2 local summary prepended to its
content slice, and an H1 region summary kept as its own retrieval unit) and binds
them into a single flat index, so that one top-k retrieval yields both fine-grained
evidence and regional context, with no hierarchical navigation and no extra
query-time model calls.

This repository contains the method code, the six baselines, the evaluation
questions, the processed per-question data, the table/figure scripts and outputs,
and one complete example book (*Far From the Madding Crowd*) with its chunking
results and judge scores.

---

## 1. Environment

* Python 3.10+ (tested on 3.12)
* The table/figure scripts under `results/` only need numpy and matplotlib.

```bash
pip install -r requirements.txt
```

Models used in the experiments (paths are injected through environment variables):

| Role | Model | Env variable |
|------|-------|--------------|
| Chunking / summaries / CR contexts | `Qwen3-4B-Instruct-2507` | `CHUNKING_MODEL_PATH` |
| Embeddings | `Qwen3-Embedding-0.6B` | `EMBEDDING_MODEL_PATH` |
| Question / answer generation (API) | `qwen3.6-plus` | `API_BASE_URL`, `DASHSCOPE_API_KEY` |
| Judging and faithfulness audit (API) | `deepseek-v4-pro` | `API_BASE_URL`, `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY` |

No credential is stored in this repository.

---

## 2. Data (BookSum)

The study uses the **test split of BookSum** (`kmfoda/booksum`): 21 works,
577 chapter-level units.

**We do not redistribute the full texts of the 21 works** (public-domain but bulky);
instead `example_data/Far From the Madding Crowd/` ships the complete artifact set
for one book, which follows exactly the same schema as the 21-work run: the
chunking output of all seven schemes, the RAPTOR nodes, and the judge scores of
the main experiment, the Contextual Retrieval baseline, and the placebo arm.

What is shipped here, and what is not:

| Shipped | Not shipped |
|---------|-------------|
| 577 evaluation questions + reference answers + verbatim evidence | full texts of the other 20 works |
| per-question 1–5 scores for all seven methods, ablation variants, five mechanism arms | raw model completions for the other 20 works |
| per-question purity / budget detail for seven methods | model weights |
| summary-unit audit verdicts (1,478 units) and question→unit map | API credentials |
| Bayesian posterior summaries | |
| complete artifact set for one example book (splits + judgments) | |

---

## 3. Repository layout

```
src/
  chunking/            HiSumChunk pipeline (noise removal, structure, H2/H1, slicing)
  retrieval/           QA build, embedding, retrieval, answer generation, judging
  metrics/             automatic metrics (ROUGE-L, BERTScore, entity)
  stats/               cluster bootstrap, ordinal regression, Bayesian ordinal model
  experiments/         ablation, mechanism (five arms + placebo), summary audit, human annotation
baselines/             Fixed-1024W, LumberChunker, LLMSemantic, SemanticEmbedding, RAPTOR, Contextual Retrieval
prompts/               every prompt text used in the paper (organized by stage)
data/
  qa/                  577 questions with answers, types, difficulty, evidence units
  scores/              per-question scores for main / ablation / mechanism, purity and budget
  slice_examples/      format examples of the two slice types (Section 5.2)
  posterior/           Bayesian and frequentist statistical results
  audit/               summary faithfulness audit verdicts and question-unit map
example_data/          one example book: scheme splits, RAPTOR nodes, judge scores
results/               table/figure scripts for tables 1-7 and figures 3-5 (+ output/)
docs/                  pseudocode and data card
```

---

## 4. Citation

```bibtex
@article{hisumchunk2026,
  title   = {HiSumChunk: Hierarchical Summary-Guided Semantic Chunking for
             Context-Preserving Long-Document RAG},
  author  = {Huang, Jiajia and Cai, Peishen and others},
  journal = {World Wide Web: Internet and Web Information Systems},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## 5. License

* Code: MIT (see `LICENSE`).
* Released data (questions, scores, audit verdicts): CC BY 4.0.
* BookSum-derived texts follow the license of `facebookresearch/booksum`; the
  example-book text comes from Project Gutenberg and is public domain.
