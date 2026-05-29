"""

Current Code Deployed on dev.mployee.me/test2

22/05/2026

What this script does: 

1. Read resume PDF from S3 and extract text
2. Use YOLO layout parsing to get headers
3. Use Groq to extract cleaned headers
4. Map YOLO headers with font sizes
5. Build sections from YOLO headers


"""

from extraction.body_content import filter_body_content_new as filter_body_content
from parsers.pdf_parser import load_local_pdf, fetch_pdf_from_s3
from groq_utils.cleaned_headers import filter_headers_with_groq
from check_multi_colsv2 import detect_columns_advanced
from extraction.section_builder import SectionBuilder
from parsers.yolo_parser import LayoutParser
# from extraction.model_inference import *
from groq_utils.prompts import *
from collections import Counter
from config.settings import *
import json
import time
import ast


MODEL_THRESHOLD = 0.9
DATA_DIR = "Check PDFs"


# -----------------------------------
# Map YOLO headers with font sizes
# -----------------------------------
def map_yolo_headers_with_fonts(yolo_blocks, word_positions):
    enriched_headers = []
    for block in yolo_blocks:
        if block["class_id"] != 7:
            continue

        header_words = set(block["text"].split())
        matched_fonts = [
            font_size
            for word, font_size, y in word_positions
            if word in header_words
        ]

        enriched_headers.append({
            "text": block["text"],
            "font": max(set(matched_fonts), key=matched_fonts.count) if matched_fonts else None,
            "y_position": block["y"]
        })
    return enriched_headers


# -----------------------------------
# Core processing logic
# -----------------------------------
def _process_pdf(pdf_bytes, pdf_path, dpi=300, conf=0.15):
    # YOLO layout parsing
    parser = LayoutParser(dpi=dpi, conf=conf)
    blocks = parser.parse(pdf_bytes)
    print(f"✅ YOLO blocks parsed: {len(blocks)}")

    # Body content + font info
    linewise_content_with_fonts, font_and_words, max_font_size, max_words, word_positions = filter_body_content(pdf_bytes)
    print(f"✅ Font and Words: {font_and_words}")

    # Map YOLO headers with fonts
    yolo_headers_with_fonts = map_yolo_headers_with_fonts(blocks, word_positions)
    print(f"✅ YOLO Headers with Fonts: {yolo_headers_with_fonts}")

    # Filter headers via Groq
    cleaned_headers, prompt_tokens, completion_tokens, total_tokens = filter_headers_with_groq(
        yolo_headers_with_fonts, cleaned_headers_prompt_template_v3
    )
    print(f"✅ Cleaned Headers: {cleaned_headers}")

    if isinstance(cleaned_headers, str):
        cleaned_headers = ast.literal_eval(cleaned_headers)

    # --- Font consistency check ---
    cleaned_headers_set = set(h.strip() for h in cleaned_headers)

    cleaned_header_objects = [
        h for h in yolo_headers_with_fonts
        if h["text"].strip() in cleaned_headers_set
    ]

    cleaned_fonts = [h["font"] for h in cleaned_header_objects if h["font"] is not None]
    font_counter = Counter(cleaned_fonts)
    max_recurring_font_size = font_counter.most_common(1)[0][0] if font_counter else None

    print(f"✅ Max Recurring Font Size (from cleaned headers): {max_recurring_font_size}")

    # Headers with dominant font missing from cleaned headers
    max_font_missing_headers = []
    if max_recurring_font_size is not None:
        max_font_missing_headers = [
            h for h in yolo_headers_with_fonts
            if h["font"] == max_recurring_font_size
            and h["text"].strip() not in cleaned_headers_set
        ]

    print(f"⚠️ Same Font Headers Missing From Cleaned Headers: {[h['text'] for h in max_font_missing_headers]}")

    # Recover missed headers via automaton
    # matched_missing = match_headers_with_automaton(max_font_missing_headers, HEADER_AUTOMATON)
    # print(f"✅ Recovered via Automaton: {matched_missing}")

    cleaned_headers.extend(h["text"] for h in max_font_missing_headers)
    print(f"✅ Final Cleaned Headers count: {len(cleaned_headers)}")

    # Detect column layout
    is_multi_column = detect_columns_advanced(pdf_path, max_pages=5).is_multicolumn
    print(f"🔍 Is Multi Column: {is_multi_column}")

    # Build sections
    builder = SectionBuilder(cleaned_headers)
    sections = builder.build(blocks, is_multi_column)
    print(f"📦 Sections: {list(sections.keys())}")

    return sections


# -----------------------------------
# Lambda handler
# -----------------------------------
def handler(event, context):
    start_time = time.time()
    try:
        print("✅ Lambda invoked")

        req_body = json.loads(event["body"]) if "body" in event else event

        aws_access_key = req_body["aws_access_key"]
        aws_secret_key = req_body["aws_secret_key"]
        pdf_url = req_body["pdf_url"]
        print(f"📥 PDF URL: {pdf_url}")
        confidence_threshold = req_body.get("confidence_threshold", 0.15)
        print(f"📊 Confidence Threshold: {confidence_threshold}")
        dpi = req_body.get("dpi", 300)
        print(f"📊 DPI: {dpi}")

        pdf_bytes, pdf_path = fetch_pdf_from_s3(pdf_url, aws_access_key, aws_secret_key)

        sections = _process_pdf(pdf_bytes, pdf_path, dpi=dpi, conf=confidence_threshold)

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            },
            # "body": json.dumps({"sections": sections})
            "body": json.dumps(sections)
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            },
            "body": json.dumps({"error": str(e)})
        }

    finally:
        print(f"⌚ Time taken: {time.time() - start_time:.2f} seconds")


# -----------------------------------
# Local testing
# -----------------------------------
if __name__ == "__main__":
    # file_name = "Eklavya_Resume26.pdf"
    file_name = "DipanshuAmrate_InternshalaResume.pdf"
    pdf_path = f"{DATA_DIR}/{file_name}"
    pdf_bytes = load_local_pdf(pdf_path)

    sections = _process_pdf(pdf_bytes, pdf_path, dpi=400, conf=0.10)
    print(json.dumps(sections, indent=2))

    with open("claude_sections.json", "w") as f:
        json.dump(sections, f, indent=2)