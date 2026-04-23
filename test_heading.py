from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, fetch_pdf_from_s3
from extraction.section_mapper import map_content_to_standard_header
from extraction.body_content import filter_body_content
from extraction.section_builder import SectionBuilder 
from extraction.headings import get_headings
from parsers.yolo_parser import LayoutParser
from groq_utils.prompts import *

import pandas as pd
import json
import os
import ast

DATA_DIR = "documents"


# -------------------------
# INPUT HANDLER 
# -------------------------
def get_pdf_bytes(input_type="local", path_or_url=None):

    if input_type == "local":
        pdf_bytes = load_local_pdf(path_or_url)
        file_name = os.path.basename(path_or_url)

    elif input_type == "s3":
        pdf_bytes, _ = fetch_pdf_from_s3(
            path_or_url,
            os.getenv("AWS_ACCESS_KEY"),
            os.getenv("AWS_SECRET_KEY")
        )
        file_name = path_or_url.split("/")[-1]

    else:
        raise ValueError("Invalid input_type. Use 'local' or 's3'")

    return pdf_bytes, file_name


# -------------------------
# YOLO Layout Parser
# -------------------------
def process_resume(pdf_bytes):
    parser = LayoutParser()
    blocks = parser.parse(pdf_bytes)
    return {"blocks": blocks}


# -------------------------
# CORE FUNCTION
# -------------------------
def analyze_single_resume(pdf_bytes, file_name="sample.pdf"):

    print(f"\n📄 Processing file: {file_name}")

    results = process_resume(pdf_bytes)

    headings, subHeadings, notRequired_Heading, Work_Project_Headings, EduSkill_Headings, Other_Headings, Other_headings_db, sectionMap, sectionMapCount, standard_match_headings, standard_match_headings_count = get_headings(pdf_bytes)

    linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_bytes)

    print(f"🔍 Extracted Headings: {subHeadings}")

    cleaned_headers = filter_headers_with_groq(
        linewise_content_with_fonts,
        cleaned_headers_prompt_template
    )

    print(f"🧹 Cleaned Headers: {cleaned_headers}")

    if isinstance(cleaned_headers, str):
        cleaned_headers = ast.literal_eval(cleaned_headers)

    standard_headings_map = get_standard_headings_map(
        cleaned_headers,
        standard_headings_prompt
        # new_standard_headings_prompt
    )

    print(f"📚 Standard Headings Map: {standard_headings_map}")

    builder = SectionBuilder(cleaned_headers)
    sections = builder.build(results["blocks"])

    standard_sections = map_content_to_standard_header(
        sections,
        standard_headings_map
    )

    # Debug dump (optional)
    with open("sections_debug.json", "w") as f:
        json.dump(sections, f, indent=4)

    # Comparison
    overlap = list(set(headings) & set(cleaned_headers))

    return {
        "file_name": file_name,
        "headings": headings,
        "subHeadings": subHeadings,
        "notRequired_Heading": notRequired_Heading,
        "standard_match_headings": standard_match_headings,
        "cleaned_headers": cleaned_headers,
        "standard_headings_map": standard_headings_map,
        "sections_keys": list(sections.keys()),
        "common_headings": overlap,
        "common_count": len(overlap),
        "yolo_heading_count": len(headings),
        "groq_heading_count": len(cleaned_headers),
    }


# -------------------------
# MULTIPLE LOCAL FILES
# -------------------------
def analyze_multiple_resumes(data_dir):

    results = []

    for file_name in os.listdir(data_dir):

        if file_name.endswith(".pdf"):

            pdf_path = os.path.join(data_dir, file_name)
            pdf_bytes, file_name = get_pdf_bytes("local", pdf_path)

            result = analyze_single_resume(pdf_bytes, file_name)
            results.append(result)

    return results


# -------------------------
# SINGLE (LOCAL OR S3)
# -------------------------
def analyze_one_resume(input_type, path_or_url):

    pdf_bytes, file_name = get_pdf_bytes(input_type, path_or_url)

    return analyze_single_resume(pdf_bytes, file_name)


# -------------------------
# SAVE
# -------------------------
def save_results_to_csv(results, file_name):

    df = pd.DataFrame(results)

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
        )

    df.to_csv(file_name, index=False)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    MODE = "batch"   # "single" or "batch"

    if MODE == "single":

        # -------- OPTION 1: LOCAL --------
        # result = analyze_one_resume("local", "documents/sample.pdf")

        # -------- OPTION 2: S3 --------
        result = analyze_one_resume(
            "s3",
            "https://local-job-match-pro.s3.ap-south-2.amazonaws.com/e9168491d6ec8e5c0fcdaced9072de5b"
        )

        save_results_to_csv([result], "single_resume_headings.csv")

        print("\n✅ Single resume analysis done")


    elif MODE == "batch":

        results = analyze_multiple_resumes(DATA_DIR)

        save_results_to_csv(results, "headings_comparison_results.csv")

        print("\n✅ Batch analysis done")