# Data card

## Overview

| Item | Value |
|---|---|
| Source dataset | BookSum (`kmfoda/booksum`), test split |
| Works | 21 (10 novels, 7 dramas, 4 nonfiction) |
| Chapter-level units | 577 |
| Evaluation questions | 577 (1 per chapter-level unit) |
| Question types | causal 430, fact 64, relation 48, event_transition 33, evidence_summary 2 |
| Evidence | verbatim fragments of the BookSum `summary_text` of the same chapter; 321 questions single-fragment, 256 multi-fragment |
| Human review | all 577 reviewed; 9 substantive revisions (1.6%) |
| Methods compared | HiSumChunk + 6 baselines |
| Judge | DeepSeek-V4-Pro, 1–5 on grounding / correctness / hallucination absence |
| Valid judgments | 4,038 (RAPTOR has 576) |

## Files

### `data/qa/qa_577.json`

One record per question:

```json
{
  "qa_id": "Far From the Madding Crowd.chapter 4.draft.qa_1",
  "book_title": "Far From the Madding Crowd",
  "chapter_id": "Far From the Madding Crowd.chapter 4",
  "question": "...",
  "answer": "...",
  "question_type": "causal",
  "difficulty": "medium",
  "evidence_units": [{"evidence_text": "...", "reason": "..."}],
  "text_source": "booksum_summary_text"
}
```

### `data/scores/scores_main_7schemes.csv`

Per-question judge scores (1-field scheme/query_id/book/question_type/difficulty +
three dimensions + any_zero flag), 4,039 rows. Rows with a zero dimension score are
excluded when computing means, exactly as in the paper.

### `data/scores/scores_ablation_4variants.csv`

Same schema for the end-to-end ablation (`wo_h1`, `wo_h2`, `wo_h1h2`, `noise_50`),
2,308 scored rows.

### `data/scores/scores_mechanism_5arms.csv`

Per-question scores for the fixed-retrieval-set arms A/B/C/D/P (2,881 rows).

### `data/scores/purity_per_question_7schemes.csv`

Per-question purity detail for Table 3 / Fig. 5:
`scheme, query_id, book, purity, total_words, max_chunk, sd, n_chapters`.

### `data/scores/five_arm_summary.json|txt`

Archived aggregate of the mechanism experiment: arm means, delta vs arm A with 95%
CIs and Holm p-values (n=575 common valid questions).

### `data/slice_examples/slice_examples.json`

Two examples of each slice type with the exact stored fields
(`region_id`, `is_region_summary`, `chunk_info`, `text`, ...).

### `data/audit/`

* `audit_verdicts.json` – 1,478 units; each with `unit_id`, `level`, `book`,
  `verdict`, `primary`, `strict` and claim-level labels.
* `question_unit_map.json` – 577 questions × top-5 slots, each with the
  `chunk_id`/`unit_id`/`level` used to flag contamination.
* `audit_summary.json|txt` – archived aggregation.

Claim-count conventions: the paper's claim-level figure (58 of 11,339, 0.51% =
42 contradicted + 16 unsupported) uses the fact-type primary criterion; the
`bad` field in `audit_summary.json` counts all claim types (66, 0.58%), adding 8
interpretation-type unsupported claims. Two minor annotation artefacts are
recorded without impact on any unit-level or Table 6 number: one malformed claim
label (`H1::The Picture of Dorian Gray::R124`, a JSON slip from the judge model
whose evidence supports "supported") and 6 zero-claim units (H2, non-retrieved,
all judged "clean, verbatim fragment").

### `data/posterior/`

Bayesian hierarchical ordinal model results for seven methods
(`bayes_results_7schemes.json`: P(better), mean logit difference, 95% HDI per
comparison; diagnostics) plus frequency-frequentist composite and ordinal results,
and the six-scheme versions.

## Known limitations (as stated in the paper)

* Questions are derived from chapter-level summaries, so they cannot cover freer
  user question styles.
* Scores are ceiling-heavy (83.0% of questions score full marks on all three
  dimensions); conclusions are about stable ranking and repeatable paired tests.
* Entity recall counts by block, which structurally favors coarse-grained indexes.
* The judge is a single model call; judging-model variance is not included in the
  confidence intervals.
