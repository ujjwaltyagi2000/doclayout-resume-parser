"""

Test script to read existing sheet of resume and headers and add new column with headings extracted using Yolo Section Headers passed to Groq (llama-3.1-8b)

"""
from groq_utils.prompts import cleaned_headers_prompt_template_v3
from groq_utils.cleaned_headers import filter_headers_with_groq
from extraction.body_content import filter_body_content_new as filter_body_content
from parsers.pdf_parser import load_local_pdf
from parsers.yolo_parser import LayoutParser
import pandas as pd
import time
import os

DATA_DIR = "documents"
INPUT_CSV = "llama-8b.csv"
OUTPUT_CSV = "headings_comparison_2404.csv"

def map_yolo_headers_with_fonts(yolo_blocks, word_positions):
    enriched_headers = []

    for block in yolo_blocks:
        if block["class_id"] != 7:
            continue

        header_text = block["text"]
        header_words = set(header_text.split())

        matched_fonts = []

        for word, font_size, y in word_positions:
            # basic word overlap check
            if word in header_words:
                matched_fonts.append(font_size)

        if matched_fonts:
            # take most common font
            font = max(set(matched_fonts), key=matched_fonts.count)
        else:
            font = None

        enriched_headers.append({
            "text": header_text,
            "font": font,
            "y_position": block["y"]
        })

    return enriched_headers

# Pass PDF bytes to YOLO layout parser and get detected blocks
def process_resume(pdf_bytes):


    parser = LayoutParser()
    blocks = parser.parse(pdf_bytes)

    return {
        "blocks": blocks,
        # "headers": headers
    }

def main():
    # Read existing CSV — all columns preserved as-is
    df = pd.read_csv(INPUT_CSV)
    print(f"📂 Loaded {len(df)} rows from {INPUT_CSV}")

    cleaned_headers_v3_results = []

    for _, row in df.iterrows():
        print(f"Sleeping for 5 seconds before processing next file...")  # To avoid hitting rate limits
        time.sleep(5)
        file_name = row["file_name"]
        pdf_path = os.path.join(DATA_DIR, file_name)

        if not os.path.exists(pdf_path):
            print(f"⚠️  File not found, skipping: {file_name}")
            cleaned_headers_v3_results.append(None)
            continue

        print(f"\n📄 Processing: {file_name}")

        pdf_bytes = load_local_pdf(pdf_path)

        results = process_resume(pdf_bytes)

        # Get linewise content (needed for Groq call)
        # linewise_content_with_fonts, _, _, _ = filter_body_content(pdf_bytes) # old code without Yolo
        linewise_content_with_fonts, _, _, _, word_positions = filter_body_content(pdf_bytes) #new code with Yolo

        # Map YOLO headers with fonts
        yolo_headers_with_fonts = map_yolo_headers_with_fonts(results["blocks"], word_positions)

        # Run model ONCE with v3 prompt
        cleaned_headers_v3 = filter_headers_with_groq(
            yolo_headers_with_fonts,
            cleaned_headers_prompt_template_v3
        )
        print(f"✅ cleaned_headers_yolo: {cleaned_headers_v3}")

        cleaned_headers_v3_results.append(cleaned_headers_v3)

    # Add new column, keep everything else untouched
    df["cleaned_headers_yolo"] = cleaned_headers_v3_results

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()