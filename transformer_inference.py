from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F



@dataclass(frozen=True)
class MapResult:
    heading: str
    label: str   # Projects | Experience | Summary | ""
    score: float
    gap: float
    ratio: float
    topk: List[Tuple[str, float]]


class PureLabelMapper:
    """
    Pure label-based semantic mapper (NO prototypes, NO keyword lists).
    Improvement: token-level scoring fallback so "Summary strength" maps to Summary.
    Returns blank "" if not confidently matching.
    """

    def __init__(
        self,
        model_path: str = "local_model",
        labels: List[str] | None = None,
        min_score: float = 0.55,
        min_gap: float = 0.10,
        min_ratio: float = 1.25,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.device = torch.device("cpu")

        # Load tokenizer + model from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        self.labels = labels or ["Projects", "Experience", "Summary"]

        self.min_score = float(min_score)
        self.min_gap = float(min_gap)
        self.min_ratio = float(min_ratio)

        # Pre-compute label embeddings
        self.label_embs = {
            lbl: self._encode(lbl.lower())
            for lbl in self.labels
        }

    @staticmethod
    def _clean(s: str) -> str:
        return " ".join((s or "").replace("\n", " ").replace("\r", " ").replace("\t", " ").strip().split())

    @lru_cache(maxsize=50000)
    def _encode(self, text: str) -> np.ndarray:

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        sentence_embedding = sum_embeddings / sum_mask

        # Normalize (like sentence-transformers does)
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        return sentence_embedding[0].cpu().numpy().astype(np.float32)

    def map_one(self, heading: str) -> MapResult:
        h = self._clean(heading)
        h_lower = h.lower()
        tokens = h_lower.split()

        # Embed full heading once
        h_emb = self._encode(h_lower)

        scored: List[Tuple[str, float]] = []
        for lbl in self.labels:
            lbl_emb = self.label_embs[lbl]

            # Score full phrase
            full_score = float(np.dot(h_emb, lbl_emb))

            # Score individual tokens (fallback for cases like "Summary strength")
            best_token_score = 0.0
            for t in tokens:
                t_emb = self._encode(t)
                best_token_score = max(best_token_score, float(np.dot(t_emb, lbl_emb)))

            # Final label score is the best of phrase vs token
            final_score = max(full_score, best_token_score)

            scored.append((lbl, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top1_lbl, top1 = scored[0]
        top2 = scored[1][1] if len(scored) > 1 else -1.0

        gap = top1 - top2
        ratio = (top1 / top2) if top2 > 0 else 999

        final = top1_lbl if (top1 >= self.min_score and gap >= self.min_gap and ratio >= self.min_ratio) else ""

        return MapResult(h, final, round(top1, 3), round(gap, 3), round(ratio, 2), scored)

    def map_many(self, headings: List[str]) -> List[MapResult]:
        return [self.map_one(h) for h in headings]


def build_final_mapping(results: List[MapResult]) -> Dict[str, object]:
    """
    Returns:
      - heading_to_label: {heading: label_or_blank}
      - grouped: {Projects:[...], Experience:[...], Summary:[...], Unmapped:[...]}
      - canonical_3: best (highest score) heading per mapped bucket
    """
    heading_to_label = {r.heading: r.label for r in results}

    grouped = defaultdict(list)
    for r in results:
        if r.label:
            grouped[r.label].append(r.heading)
        else:
            grouped["Unmapped"].append(r.heading)

    for k in ["Projects", "Experience", "Summary", "Unmapped"]:
        grouped.setdefault(k, [])

    canonical_3 = {"Projects": None, "Experience": None, "Summary": None}
    best_score = {"Projects": float("-inf"), "Experience": float("-inf"), "Summary": float("-inf")}
    for r in results:
        if r.label in best_score and r.score > best_score[r.label]:
            best_score[r.label] = r.score
            canonical_3[r.label] = r.heading

    return {
        "heading_to_label": heading_to_label,
        "grouped": dict(grouped),
        "canonical_3": canonical_3,
    }

def run_transformer_mapping(headings: List[str]) -> dict:

    mapper = PureLabelMapper(
        min_score=0.55,
        min_gap=0.10,
        min_ratio=1.25
    )


    results = mapper.map_many(headings)

    print("\n=== Per-heading result ===")
    for r in results:
        print(f"{r.heading:30s} -> {r.label:10s} score={r.score} gap={r.gap} ratio={r.ratio} top3={r.topk[:3]}")

    final = build_final_mapping(results)

    print("\n=== Final heading_to_label mapping ===")
    for h, lbl in final["heading_to_label"].items():
        print(f"{h:30s} -> {lbl}")

    print("\n=== Grouped final mapping ===")
    for k in ["Experience", "Projects", "Summary", "Unmapped"]:
        print(f"{k}: {final['grouped'][k]}")

    print("\n=== Canonical 3 (best per bucket) ===")
    print(final["canonical_3"])
    return final["canonical_3"]

if __name__ == "__main__":
    sections = [
        "Education",
        "Work Experience",
        "Technical Skills",
        "Certifications and Trainings",
        "Projects",
        "Summary strength",
        "Professional Journey",
        "What I Built",
    ]

    headers_map = run_transformer_mapping(sections)
    print(type(headers_map))