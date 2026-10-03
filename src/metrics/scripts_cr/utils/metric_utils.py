from __future__ import annotations

from typing import Any

from rouge_score import rouge_scorer

from config import ROUGE_USE_STEMMER
from utils.io_utils import ProgressLogger, log_message


class RougeLScorer:
    def __init__(self, use_stemmer: bool = ROUGE_USE_STEMMER) -> None:
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)

    def rouge_l_f1(self, reference: str, prediction: str) -> float | None:
        if not reference.strip() or not prediction.strip():
            return None
        return float(self.scorer.score(reference, prediction)["rougeL"].fmeasure)


_BERT_SCORER_CACHE: dict[tuple[str, str, bool, str | None, int], Any] = {}


def get_bertscorer(
    model_type: str,
    lang: str,
    batch_size: int,
    rescale_with_baseline: bool,
    device: str | None,
) -> Any:
    cache_key = (model_type, lang, rescale_with_baseline, device, batch_size)
    if cache_key not in _BERT_SCORER_CACHE:
        from bert_score import BERTScorer

        log_message(f"Loading BERTScorer once | model={model_type} | lang={lang} | batch_size={batch_size} | device={device or 'auto'}")
        kwargs: dict[str, Any] = {
            "model_type": model_type,
            "lang": lang,
            "batch_size": batch_size,
            "rescale_with_baseline": rescale_with_baseline,
        }
        if model_type.startswith("/") and "roberta-base" in model_type:
            kwargs["num_layers"] = 10
        if device:
            kwargs["device"] = device
        _BERT_SCORER_CACHE[cache_key] = BERTScorer(**kwargs)
        log_message(f"BERTScorer loaded | model={model_type}")
    return _BERT_SCORER_CACHE[cache_key]


def compute_bertscore_f1(
    predictions: list[str],
    references: list[str],
    model_type: str,
    lang: str,
    batch_size: int,
    rescale_with_baseline: bool,
    device: str | None,
    progress_name: str = "BERTScore",
) -> list[float | None]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    results: list[float | None] = [None] * len(predictions)
    valid_indices: list[int] = []
    valid_predictions: list[str] = []
    valid_references: list[str] = []

    filter_progress = ProgressLogger(f"{progress_name} filter valid pairs", total=len(predictions), every=max(len(predictions) // 20, 1))
    for idx, (prediction, reference) in enumerate(zip(predictions, references), start=1):
        if prediction.strip() and reference.strip():
            valid_indices.append(idx - 1)
            valid_predictions.append(prediction)
            valid_references.append(reference)
        filter_progress.update(idx, detail=f"valid={len(valid_indices)}")
    filter_progress.finish(detail=f"valid_pairs={len(valid_indices)}")

    if not valid_indices:
        log_message(f"{progress_name}: no valid pairs; skip BERTScore computation")
        return results

    scorer = get_bertscorer(
        model_type=model_type,
        lang=lang,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
        device=device,
    )
    total_batches = (len(valid_predictions) + batch_size - 1) // batch_size
    batch_progress = ProgressLogger(f"{progress_name} batches", total=total_batches, every=1)

    for batch_index, start in enumerate(range(0, len(valid_predictions), batch_size), start=1):
        end = min(start + batch_size, len(valid_predictions))
        _, _, f1_values = scorer.score(
            valid_predictions[start:end],
            valid_references[start:end],
            verbose=False,
            batch_size=batch_size,
        )
        for original_idx, f1 in zip(valid_indices[start:end], f1_values.tolist()):
            results[original_idx] = float(f1)
        batch_progress.update(batch_index, detail=f"pairs={end}/{len(valid_predictions)}")
    batch_progress.finish(detail=f"valid_pairs={len(valid_predictions)}")
    return results


def max_sentence_scores(score_matrix_rows: list[list[float | None]]) -> list[float | None]:
    max_scores: list[float | None] = []
    for row in score_matrix_rows:
        valid = [score for score in row if score is not None]
        max_scores.append(max(valid) if valid else None)
    return max_scores
