from __future__ import annotations

import re
from typing import Any

from config import ENTITY_LABELS, SPACY_MODEL
from utils.io_utils import f1_score, normalize_space


def load_spacy_model(model_name: str = SPACY_MODEL) -> Any:
    import spacy

    return spacy.load(model_name)


def normalize_entity(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    text = re.sub(r"\b(the|a|an)\b", "", text)
    return normalize_space(text)


def extract_entities(text: str, nlp: Any, labels: list[str] = ENTITY_LABELS) -> list[str]:
    if not text.strip():
        return []
    doc = nlp(text)
    entities: list[str] = []
    allowed = set(labels)
    for ent in doc.ents:
        if ent.label_ not in allowed:
            continue
        normalized = normalize_entity(ent.text)
        if normalized and normalized not in entities:
            entities.append(normalized)
    return entities


def entity_overlap_metrics(pred_entities: list[str], ref_entities: list[str]) -> dict[str, float | None]:
    pred_set = set(pred_entities)
    ref_set = set(ref_entities)
    if not pred_set and not ref_set:
        return {"precision": None, "recall": None, "f1": None, "overlap_count": 0}
    overlap_count = len(pred_set & ref_set)
    precision = overlap_count / len(pred_set) if pred_set else 0.0
    recall = overlap_count / len(ref_set) if ref_set else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "overlap_count": overlap_count,
    }


def entity_support_metrics(answer_entities: list[str], context_entities: list[str]) -> dict[str, float | None]:
    answer_set = set(answer_entities)
    context_set = set(context_entities)
    if not answer_set:
        return {
            "unsupported_entity_rate": None,
            "entity_support_rate": None,
            "supported_entity_count": 0,
            "unsupported_entity_count": 0,
        }
    supported = answer_set & context_set
    unsupported = answer_set - context_set
    unsupported_rate = len(unsupported) / len(answer_set)
    support_rate = len(supported) / len(answer_set)
    return {
        "unsupported_entity_rate": unsupported_rate,
        "entity_support_rate": support_rate,
        "supported_entity_count": len(supported),
        "unsupported_entity_count": len(unsupported),
    }
