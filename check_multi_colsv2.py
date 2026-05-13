from __future__ import annotations

import fitz
import pandas as pd

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIG
# ============================================================

# FOLDER_PATH = "documents"
FOLDER_PATH = "Check PDFs"
OUTPUT_EXCEL = "output.xlsx"


# ============================================================
# CODE 1: SIMPLE DETECTOR
# ============================================================

def detect_multicolumn_simple(page, min_gap_ratio=0.06):
    words = page.get_text("words")

    if not words:
        return {
            "is_multicolumn": False,
            "columns": 1,
            "confidence": 0.0,
            "reason": "No words found",
            "gap": None,
            "word_count": 0,
        }

    page_width = page.rect.width

    min_x = min(w[0] for w in words)
    max_x = max(w[2] for w in words)
    text_width = max_x - min_x

    if text_width <= 0:
        return {
            "is_multicolumn": False,
            "columns": 1,
            "confidence": 0.0,
            "reason": "Invalid text width",
            "gap": None,
            "word_count": len(words),
        }

    density = [0] * int(page_width + 1)

    for w in words:
        x0, y0, x1, y1 = w[:4]

        for x in range(int(x0), min(int(x1) + 1, len(density))):
            density[x] += 1

    start = int(min_x)
    end = int(max_x)

    min_gap_width = text_width * min_gap_ratio

    gaps = []
    in_gap = False
    gap_start = None

    for x in range(start, end):
        if density[x] == 0 and not in_gap:
            in_gap = True
            gap_start = x

        elif density[x] > 0 and in_gap:
            in_gap = False
            gap_end = x
            gap_width = gap_end - gap_start

            if gap_width >= min_gap_width:
                gaps.append((gap_start, gap_end, gap_width))

    if in_gap and gap_start is not None:
        gap_end = end
        gap_width = gap_end - gap_start

        if gap_width >= min_gap_width:
            gaps.append((gap_start, gap_end, gap_width))

    middle_gaps = []

    for gs, ge, gw in gaps:
        gap_center = (gs + ge) / 2

        if min_x + text_width * 0.25 <= gap_center <= min_x + text_width * 0.75:
            middle_gaps.append((gs, ge, gw))

    if middle_gaps:
        largest_gap = max(middle_gaps, key=lambda g: g[2])

        return {
            "is_multicolumn": True,
            "columns": 2,
            "confidence": 0.80,
            "reason": "Large empty vertical gap found between text areas",
            "gap": largest_gap,
            "word_count": len(words),
        }

    return {
        "is_multicolumn": False,
        "columns": 1,
        "confidence": 0.75,
        "reason": "No strong middle vertical gap found",
        "gap": None,
        "word_count": len(words),
    }


def detect_resume_columns_simple(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            page_results = []

            for i, page in enumerate(doc):
                result = detect_multicolumn_simple(page)
                result["page"] = i + 1
                page_results.append(result)

        multicol_pages = [
            r["page"] for r in page_results
            if r["is_multicolumn"]
        ]

        total_words = sum(r.get("word_count", 0) for r in page_results)

        return {
            "status": "success",
            "pdf_path": pdf_path,
            "total_pages": len(page_results),
            "total_words": total_words,
            "is_multicolumn_resume": len(multicol_pages) > 0,
            "layout": "double_column" if len(multicol_pages) > 0 else "single_column",
            "confidence": 0.80 if len(multicol_pages) > 0 else 0.75,
            "multicolumn_pages": multicol_pages,
            "page_results": page_results,
            "error": None,
        }

    except Exception as exc:
        return {
            "status": "error",
            "pdf_path": pdf_path,
            "total_pages": 0,
            "total_words": 0,
            "is_multicolumn_resume": None,
            "layout": "unknown",
            "confidence": 0.0,
            "multicolumn_pages": [],
            "page_results": [],
            "error": repr(exc),
        }


# ============================================================
# CODE 2: ADVANCED DETECTOR
# ============================================================

@dataclass
class GapInfo:
    x_start: float
    x_end: float
    width: float
    center_ratio: float
    vertical_consistency: float


@dataclass
class PageResult:
    page: int
    is_multicolumn: bool
    columns: int
    confidence: float
    reason: str
    gap: Optional[GapInfo] = None
    text_x_range: Tuple[float, float] = (0.0, 0.0)
    word_count: int = 0


@dataclass
class DocumentResult:
    pdf_path: str
    total_pages: int
    is_multicolumn: Optional[bool]
    confidence: float
    status: str = "success"
    layout: str = "unknown"
    multicolumn_pages: List[int] = field(default_factory=list)
    page_results: List[PageResult] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_header_bottom(words: list, page_height: float) -> float:
    if not words:
        return 0.0

    top_cap = page_height * 0.30
    bands: Dict[int, List[Tuple[float, float]]] = {}

    for w in words:
        bucket = int(w[1] / 8)
        bands.setdefault(bucket, []).append((float(w[0]), float(w[2])))

    if not bands:
        return 0.0

    min_x = min(float(w[0]) for w in words)
    max_x = max(float(w[2]) for w in words)
    page_span = max_x - min_x

    if page_span <= 0:
        return 0.0

    for bucket in sorted(bands):
        xs = bands[bucket]
        band_span = max(x1 for _, x1 in xs) - min(x0 for x0, _ in xs)

        if band_span < page_span * 0.60:
            return min(bucket * 8.0, top_cap)

    return 0.0


def _build_density_fast(
    words: list,
    page_width: float,
    y_start: float,
    sparsity_threshold: int = 1,
) -> List[int]:
    size = max(2, int(page_width) + 2)
    diff = [0] * (size + 1)

    for w in words:
        x0, y0, x1, _y1 = w[:4]

        if float(y0) < y_start:
            continue

        start = max(0, int(x0))
        end = min(size - 1, int(x1))

        if start <= end:
            diff[start] += 1
            diff[end + 1] -= 1

    density = [0] * size
    running = 0

    for i in range(size):
        running += diff[i]
        density[i] = 0 if running <= sparsity_threshold else running

    return density


def _vertical_gap_consistency(
    words: list,
    gap_x0: float,
    gap_x1: float,
    y_start: float,
    page_height: float,
    slices: int = 10,
) -> float:
    if slices <= 0 or page_height <= y_start:
        return 0.0

    slice_height = (page_height - y_start) / slices
    empty_slices = 0

    for i in range(slices):
        y_lo = y_start + i * slice_height
        y_hi = y_lo + slice_height
        has_word_in_gap = False

        for w in words:
            x0, y0, x1, _y1 = w[:4]

            if float(y0) < y_start:
                continue

            word_top_in_slice = y_lo <= float(y0) <= y_hi
            word_overlaps_gap = not (
                float(x1) <= gap_x0 or float(x0) >= gap_x1
            )

            if word_top_in_slice and word_overlaps_gap:
                has_word_in_gap = True
                break

        if not has_word_in_gap:
            empty_slices += 1

    return empty_slices / slices


def _score_gap(
    gap_width: float,
    text_width: float,
    consistency: float,
) -> float:
    width_ratio = gap_width / max(text_width, 1.0)
    width_score = min(1.0, width_ratio / 0.15)
    score = 0.40 * width_score + 0.60 * consistency
    return round(score, 3)


def detect_multicolumn_page_advanced(
    page: fitz.Page,
    min_gap_ratio: float = 0.05,
    center_band: Tuple[float, float] = (0.20, 0.80),
    sparsity_threshold: int = 1,
    min_vertical_consistency: float = 0.55,
) -> PageResult:
    words = page.get_text("words")
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    if not words:
        return PageResult(
            page=0,
            is_multicolumn=False,
            columns=1,
            confidence=0.0,
            reason="No words found. PDF may be scanned/image-only.",
            word_count=0,
        )

    min_x = min(float(w[0]) for w in words)
    max_x = max(float(w[2]) for w in words)
    text_width = max_x - min_x

    if text_width <= 0:
        return PageResult(
            page=0,
            is_multicolumn=False,
            columns=1,
            confidence=0.0,
            reason="Invalid text width.",
            word_count=len(words),
        )

    header_bottom = _find_header_bottom(words, page_height)

    density = _build_density_fast(
        words=words,
        page_width=page_width,
        y_start=header_bottom,
        sparsity_threshold=sparsity_threshold,
    )

    min_gap_width = text_width * min_gap_ratio
    gaps: List[GapInfo] = []

    in_gap = False
    gap_start = 0

    scan_start = max(0, int(min_x))
    scan_end = min(int(max_x) + 1, len(density))

    for x in range(scan_start, scan_end):
        is_empty = density[x] == 0

        if is_empty and not in_gap:
            in_gap = True
            gap_start = x

        elif not is_empty and in_gap:
            in_gap = False
            gap_end = x
            gap_width = gap_end - gap_start

            if gap_width < min_gap_width:
                continue

            gap_center = (gap_start + gap_end) / 2.0
            center_ratio = (gap_center - min_x) / text_width

            if not (center_band[0] <= center_ratio <= center_band[1]):
                continue

            consistency = _vertical_gap_consistency(
                words=words,
                gap_x0=float(gap_start),
                gap_x1=float(gap_end),
                y_start=header_bottom,
                page_height=page_height,
            )

            gaps.append(
                GapInfo(
                    x_start=float(gap_start),
                    x_end=float(gap_end),
                    width=float(gap_width),
                    center_ratio=round(center_ratio, 3),
                    vertical_consistency=round(consistency, 3),
                )
            )

    if in_gap:
        gap_end = scan_end
        gap_width = gap_end - gap_start

        if gap_width >= min_gap_width:
            gap_center = (gap_start + gap_end) / 2.0
            center_ratio = (gap_center - min_x) / text_width

            if center_band[0] <= center_ratio <= center_band[1]:
                consistency = _vertical_gap_consistency(
                    words=words,
                    gap_x0=float(gap_start),
                    gap_x1=float(gap_end),
                    y_start=header_bottom,
                    page_height=page_height,
                )

                gaps.append(
                    GapInfo(
                        x_start=float(gap_start),
                        x_end=float(gap_end),
                        width=float(gap_width),
                        center_ratio=round(center_ratio, 3),
                        vertical_consistency=round(consistency, 3),
                    )
                )

    strong_gaps = [
        g for g in gaps
        if g.vertical_consistency >= min_vertical_consistency
    ]

    if not strong_gaps:
        borderline = [
            g for g in gaps
            if g.vertical_consistency >= 0.30
        ]

        confidence = 0.85 if not borderline else 0.65

        reason = (
            "No strong vertical gap found."
            if not borderline
            else "Gaps found but they lack vertical consistency."
        )

        return PageResult(
            page=0,
            is_multicolumn=False,
            columns=1,
            confidence=confidence,
            reason=reason,
            text_x_range=(round(min_x, 2), round(max_x, 2)),
            word_count=len(words),
        )

    best_gap = max(
        strong_gaps,
        key=lambda g: _score_gap(
            gap_width=g.width,
            text_width=text_width,
            consistency=g.vertical_consistency,
        ),
    )

    confidence = _score_gap(
        gap_width=best_gap.width,
        text_width=text_width,
        consistency=best_gap.vertical_consistency,
    )

    return PageResult(
        page=0,
        is_multicolumn=True,
        columns=2,
        confidence=confidence,
        reason=(
            f"Column separator gap detected "
            f"({best_gap.width:.0f}pt wide, "
            f"{best_gap.vertical_consistency * 100:.0f}% vertically consistent)."
        ),
        gap=best_gap,
        text_x_range=(round(min_x, 2), round(max_x, 2)),
        word_count=len(words),
    )


def detect_columns_advanced(
    pdf_path: str,
    min_gap_ratio: float = 0.05,
    min_confidence: float = 0.55,
    max_pages: Optional[int] = 5,
) -> DocumentResult:
    path = Path(pdf_path)

    if not path.exists():
        return DocumentResult(
            pdf_path=str(path),
            total_pages=0,
            is_multicolumn=None,
            confidence=0.0,
            status="error",
            layout="unknown",
            error="File does not exist.",
        )

    page_results: List[PageResult] = []

    try:
        with fitz.open(str(path)) as doc:
            actual_total_pages = len(doc)

            if max_pages is None:
                pages_to_process = actual_total_pages
            else:
                pages_to_process = min(actual_total_pages, max_pages)

            for i in range(pages_to_process):
                page = doc[i]

                result = detect_multicolumn_page_advanced(
                    page=page,
                    min_gap_ratio=min_gap_ratio,
                )

                result.page = i + 1
                page_results.append(result)

    except Exception as exc:
        return DocumentResult(
            pdf_path=str(path),
            total_pages=len(page_results),
            is_multicolumn=None,
            confidence=0.0,
            status="error",
            layout="unknown",
            page_results=page_results,
            error=repr(exc),
        )

    total_words = sum(r.word_count for r in page_results)

    if total_words == 0:
        return DocumentResult(
            pdf_path=str(path),
            total_pages=len(page_results),
            is_multicolumn=None,
            confidence=0.0,
            status="no_text",
            layout="unknown",
            page_results=page_results,
            error="No extractable text found. PDF may be scanned/image-only.",
        )

    content_pages = [
        r for r in page_results
        if r.word_count >= 30
    ]

    if not content_pages:
        return DocumentResult(
            pdf_path=str(path),
            total_pages=len(page_results),
            is_multicolumn=None,
            confidence=0.0,
            status="low_confidence",
            layout="unknown",
            page_results=page_results,
            error="Not enough text content to determine layout.",
        )

    multicolumn_votes = 0.0
    single_votes = 0.0

    for r in content_pages:
        if r.confidence < min_confidence:
            continue

        weight = 2.0 if r.page == 1 else 1.0

        if r.is_multicolumn:
            multicolumn_votes += weight * r.confidence
        else:
            single_votes += weight * r.confidence

    total_vote = multicolumn_votes + single_votes

    if total_vote == 0:
        return DocumentResult(
            pdf_path=str(path),
            total_pages=len(page_results),
            is_multicolumn=None,
            confidence=0.0,
            status="low_confidence",
            layout="unknown",
            page_results=page_results,
            error="No page had enough confidence to vote.",
        )

    multicolumn_ratio = multicolumn_votes / total_vote
    single_ratio = single_votes / total_vote

    if multicolumn_ratio >= 0.60:
        is_multicolumn: Optional[bool] = True
        layout = "double_column"
        confidence = round(multicolumn_ratio, 3)
        status = "success"

    elif single_ratio >= 0.60:
        is_multicolumn = False
        layout = "single_column"
        confidence = round(single_ratio, 3)
        status = "success"

    else:
        is_multicolumn = None
        layout = "unknown"
        confidence = round(max(multicolumn_ratio, single_ratio), 3)
        status = "low_confidence"

    multicolumn_pages = [
        r.page for r in page_results
        if r.is_multicolumn
    ]

    return DocumentResult(
        pdf_path=str(path),
        total_pages=len(page_results),
        is_multicolumn=is_multicolumn,
        confidence=confidence,
        status=status,
        layout=layout,
        multicolumn_pages=multicolumn_pages,
        page_results=page_results,
    )


# ============================================================
# EXCEL HELPERS
# ============================================================

def pages_to_string(pages):
    if not pages:
        return ""
    return ", ".join(str(p) for p in pages)


def simple_page_summary(simple_result):
    page_results = simple_result.get("page_results", [])
    parts = []

    for r in page_results:
        parts.append(
            f"Page {r.get('page')}: "
            f"multi={r.get('is_multicolumn')}, "
            f"conf={r.get('confidence')}, "
            f"words={r.get('word_count')}, "
            f"reason={r.get('reason')}"
        )

    return " | ".join(parts)


def advanced_page_summary(advanced_result):
    page_results = advanced_result.page_results
    parts = []

    for r in page_results:
        parts.append(
            f"Page {r.page}: "
            f"multi={r.is_multicolumn}, "
            f"conf={r.confidence}, "
            f"words={r.word_count}, "
            f"reason={r.reason}"
        )

    return " | ".join(parts)


# ============================================================
# RUN BOTH CODES ON FOLDER
# ============================================================

def run_both_detectors_on_folder(
    folder_path: str,
    output_excel_path: str,
    max_pages: Optional[int] = 5,
):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in folder: {folder_path}")

    rows = []

    total = len(pdf_files)

    for idx, pdf_file in enumerate(pdf_files, start=1):
        print(f"[{idx}/{total}] Processing: {pdf_file.name}")

        pdf_path = str(pdf_file)

        simple_result = detect_resume_columns_simple(pdf_path)
        advanced_result = detect_columns_advanced(
            pdf_path,
            max_pages=max_pages,
        )

        simple_is_multi = simple_result.get("is_multicolumn_resume")
        advanced_is_multi = advanced_result.is_multicolumn

        if simple_is_multi == advanced_is_multi:
            agreement = "same"
        else:
            agreement = "different"

        row = {
            "file_name": pdf_file.name,
            "file_path": pdf_path,

            "code1_status": simple_result.get("status"),
            "code1_layout": simple_result.get("layout"),
            "code1_is_multicolumn": simple_result.get("is_multicolumn_resume"),
            "code1_confidence": simple_result.get("confidence"),
            "code1_total_pages": simple_result.get("total_pages"),
            "code1_total_words": simple_result.get("total_words"),
            "code1_multicolumn_pages": pages_to_string(simple_result.get("multicolumn_pages")),
            "code1_error": simple_result.get("error"),
            "code1_page_summary": simple_page_summary(simple_result),

            "code2_status": advanced_result.status,
            "code2_layout": advanced_result.layout,
            "code2_is_multicolumn": advanced_result.is_multicolumn,
            "code2_confidence": advanced_result.confidence,
            "code2_total_pages_read": advanced_result.total_pages,
            "code2_multicolumn_pages": pages_to_string(advanced_result.multicolumn_pages),
            "code2_error": advanced_result.error,
            "code2_page_summary": advanced_page_summary(advanced_result),

            "both_codes_agreement": agreement,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    summary_rows = [
        {
            "metric": "total_pdfs",
            "value": len(df),
        },
        {
            "metric": "code1_double_column_count",
            "value": int((df["code1_is_multicolumn"] == True).sum()),
        },
        {
            "metric": "code1_single_column_count",
            "value": int((df["code1_is_multicolumn"] == False).sum()),
        },
        {
            "metric": "code2_double_column_count",
            "value": int((df["code2_is_multicolumn"] == True).sum()),
        },
        {
            "metric": "code2_single_column_count",
            "value": int((df["code2_is_multicolumn"] == False).sum()),
        },
        {
            "metric": "code2_unknown_count",
            "value": int(df["code2_is_multicolumn"].isna().sum()),
        },
        {
            "metric": "same_result_count",
            "value": int((df["both_codes_agreement"] == "same").sum()),
        },
        {
            "metric": "different_result_count",
            "value": int((df["both_codes_agreement"] == "different").sum()),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    output_path = Path(output_excel_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

        workbook = writer.book

        results_sheet = writer.sheets["Results"]
        summary_sheet = writer.sheets["Summary"]

        for sheet in [results_sheet, summary_sheet]:
            sheet.freeze_panes = "A2"

            for column_cells in sheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    try:
                        cell_value = "" if cell.value is None else str(cell.value)
                        max_length = max(max_length, len(cell_value))
                    except Exception:
                        pass

                adjusted_width = min(max(max_length + 2, 12), 60)
                sheet.column_dimensions[column_letter].width = adjusted_width

        for cell in results_sheet[1]:
            cell.font = cell.font.copy(bold=True)

        for cell in summary_sheet[1]:
            cell.font = cell.font.copy(bold=True)

    print("\nDone.")
    print(f"PDF files processed: {len(df)}")
    print(f"Excel saved at: {output_path}")

    return df, summary_df


# ============================================================
# EXECUTE
# ============================================================

# results_df, summary_df = run_both_detectors_on_folder(
#     folder_path=FOLDER_PATH,
#     output_excel_path=OUTPUT_EXCEL,
#     max_pages=5,
# )

# summary_df


# pdf_path = "documents/Ajay Data Analyst.pdf"
# pdf_path = "documents/Vinay_P_12042026 (1).pdf"
# pdf_path = "documents/Aagam_Shah_Resume.pdf"
# simple_result = detect_resume_columns_simple(pdf_path)
# print(simple_result)

# advanced_result = detect_columns_advanced(
#     pdf_path,
#     max_pages=5,
# )
# print(advanced_result.is_multicolumn)

