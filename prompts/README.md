# Prompt inventory

Every prompt text used in the paper, organized by stage. All prompts are released
verbatim as they were used in the experiments.

| Directory | Stage | Files |
|---|---|---|
| `chunking_pipeline_inline_prompts.json` | Chunking (structure detection, noise removal, H2 outlines, region merge, H1 outlines) | inline prompt templates extracted from `src/chunking/*.py` |
| `qa_build/` | Question generation (English final version) | `qa_generation_system.txt`, `qa_generation_user.txt` |
| `qa_build_full/` | Full QA-construction toolchain (original working set, includes Chinese-authored variants and evidence-check / rewrite prompts) | 9 files |
| `retrieval/` | Answer generation + LLM-as-Judge (six-scheme main experiment) | `answer_generation_*.txt`, `judge_system.txt`, `judge_user.txt` |
| `contextual_retrieval/` | CR baseline: locating-description prompt + generation/judging | `cr_context_user.txt` + generation/judging |
| `ablation/` | End-to-end ablation answering/judging | 4 files |
| `mechanism/` | Fixed-retrieval-set arms A/B/C/D | 4 files |
| `placebo/` | Placebo arm P | 4 files |
| `summary_audit/` | Claim-level faithfulness audit | `audit_system.v3.txt`, `audit_user.v3.txt` |

Model roles: chunking/summaries/CR contexts use the local `Qwen3-4B-Instruct-2507`;
question generation and answer generation use `qwen3.6-plus`; LLM-as-Judge and the
audit use `deepseek-v4-pro`.
