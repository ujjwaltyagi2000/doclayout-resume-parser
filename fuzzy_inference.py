from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict
from thefuzz import process, fuzz


STANDARD_LABELS = ["Projects", "Experience", "Summary", "Objective"]


@dataclass
class FuzzyMapResult:
    heading: str
    label: str
    score: int


def map_one_heading(
    heading: str,
    labels: List[str],
    threshold: int = 75
) -> FuzzyMapResult:
    """
    Maps a single heading using fuzzy matching.
    Returns blank label if score < threshold.
    """

    best_match, score = process.extractOne(
        heading,
        labels,
        scorer=fuzz.token_sort_ratio
    )

    final_label = best_match if score >= threshold else ""

    return FuzzyMapResult(
        heading=heading,
        label=final_label,
        score=score
    )


def map_many_headings(
    headings: List[str],
    labels: List[str] = None,
    threshold: int = 75
) -> Dict[str, object]:

    labels = labels or STANDARD_LABELS

    results: List[FuzzyMapResult] = [
        map_one_heading(h, labels, threshold)
        for h in headings
    ]

    heading_to_label = {r.heading: r.label for r in results}

    grouped = defaultdict(list)
    for r in results:
        if r.label:
            grouped[r.label].append(r.heading)
        else:
            grouped["Unmapped"].append(r.heading)

    for lbl in labels:
        grouped.setdefault(lbl, [])
    grouped.setdefault("Unmapped", [])

    canonical = {lbl: None for lbl in labels}
    best_score = {lbl: -1 for lbl in labels}

    for r in results:
        if r.label and r.score > best_score[r.label]:
            best_score[r.label] = r.score
            canonical[r.label] = r.heading

    return {
        "heading_to_label": heading_to_label,
        "grouped": dict(grouped),
        "canonical": canonical
    }

def run_fuzzy_mapping(headings: List[str]) -> dict:
    return map_many_headings(headings)

if __name__ == "__main__":

    sample_headings = [
        "Education",
        "Work Experience",
        "Technical Skills",
        "Certifications and Trainings",
        "Projects",
        "Summary strength",
        "Professional Journey",
        "What I Built",
    ]

    output = run_fuzzy_mapping(sample_headings)

    print("\n=== Fuzzy Mapping Output ===")
    print(output)