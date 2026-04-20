from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, extract_full_text
from extraction.section_mapper import map_content_to_standard_header
from extraction.body_content import filter_body_content
from extraction.section_builder import SectionBuilder 
from extraction.pointers import *
from extraction.headings import get_headings
from parsers.yolo_parser import LayoutParser
from groq_utils.prompts import *

from modules.information import *
from config.settings import *

import json
import os
import ast
import pandas as pd

DATA_DIR = "docs"

# -------------------------
# STEP 1: YOLO Layout Parser
# -------------------------
def process_resume(pdf_bytes):
    parser = LayoutParser()
    blocks = parser.parse(pdf_bytes)

    return {
        "blocks": blocks
    }


# -------------------------
# STEP 2: Metrics Function
# -------------------------
def compute_action_metrics(text):

    action_words, total_action_words, all_action_words = get_action_words(text)

    frequency_list, total_frequent_action_words, repeated_frequency = frequency_Action_words(all_action_words)

    negative_action_words, total_negative_action_words, all_negative_action_words = get_negative_action_words(text)

    frequencyList_negative, total_repeated_actionwords_negative, repeated_frequency_negative = frequency_Action_words(all_negative_action_words)

    return {
        "action_words": action_words,
        "total_action_words": total_action_words,
        "all_action_words": all_action_words,
        "frequent_action_words": frequency_list,
        "total_frequent_action_words": total_frequent_action_words,
        "repeated_frequency": repeated_frequency,
        "negative_action_words": negative_action_words,
        "total_negative_action_words": total_negative_action_words,
        "all_negative_action_words": all_negative_action_words,
        "frequencyList_negative": frequencyList_negative,
        "total_repeated_actionwords_negative": total_repeated_actionwords_negative,
        "repeated_frequency_negative": repeated_frequency_negative
    }


# -------------------------
# MAIN FUNCTION
# -------------------------
def compare_action_words():

    analysis_results = []

    for file_name in os.listdir(DATA_DIR):

        if file_name.endswith(".pdf"):

            print(f"\n📄 Processing file: {file_name}")

            pdf_path = os.path.join(DATA_DIR, file_name)
            pdf_bytes = load_local_pdf(pdf_path)

            # -------------------------
            # STEP 1: Layout Parsing
            # -------------------------
            results = process_resume(pdf_bytes)

            # -------------------------
            # STEP 2: Extract Headings
            # -------------------------
            headings, subHeadings, notRequired_Heading, Work_Project_Headings, EduSkill_Headings, Other_Headings, Other_headings_db, sectionMap, sectionMapCount, standard_match_headings, standard_match_headings_count = get_headings(pdf_bytes)

            # -------------------------
            # STEP 3: Body Content
            # -------------------------
            linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_bytes)

            # -------------------------
            # STEP 4: Clean Headers (Groq)
            # -------------------------
            cleaned_headers = filter_headers_with_groq(linewise_content_with_fonts, cleaned_headers_prompt_template)

            if isinstance(cleaned_headers, str):
                cleaned_headers = ast.literal_eval(cleaned_headers)

            print(f"✅ Cleaned Headers: {cleaned_headers}")

            # -------------------------
            # STEP 5: Standard Headings Map
            # -------------------------
            standard_headings_map = get_standard_headings_map(cleaned_headers, standard_headings_prompt)

            print(f"✅ Standard Headings Map: {standard_headings_map}")

            # -------------------------
            # STEP 6: Section Building
            # -------------------------
            builder = SectionBuilder(cleaned_headers)
            sections = builder.build(results["blocks"])

            with open(SECTIONS_OUTPUT_FILE_PATH, "w") as f:
                json.dump(sections, f, indent=4)

            # -------------------------
            # STEP 7: Map to Standard Sections
            # -------------------------
            standard_sections = map_content_to_standard_header(sections, standard_headings_map)

            # -------------------------
            # CASE 1: Experience + Projects
            # -------------------------
            exp_data = standard_sections.get("Experience", {})
            proj_data = standard_sections.get("Projects", {})

            exp_proj_text = (
                exp_data.get("text", "") + " " + proj_data.get("text", "")
            ).strip()

            exp_proj_metrics = compute_action_metrics(exp_proj_text)

            # -------------------------
            # CASE 2: Full Resume
            # -------------------------
            full_resume_text = extract_full_text(pdf_bytes)

            full_resume_metrics = compute_action_metrics(full_resume_text)

            # -------------------------
            # RESULT DOC (flattened)
            # -------------------------
            result_doc = {
                "file_name": file_name,

                # -------- EXP + PROJECT --------
                "exp_proj_action_words": exp_proj_metrics["action_words"],
                "exp_proj_all_action_words": exp_proj_metrics["all_action_words"],
                "exp_proj_negative_action_words": exp_proj_metrics["negative_action_words"],
                "exp_proj_all_negative_action_words": exp_proj_metrics["all_negative_action_words"],

                # -------- FULL RESUME --------
                "full_action_words": full_resume_metrics["action_words"],
                "full_all_action_words": full_resume_metrics["all_action_words"],
                "full_negative_action_words": full_resume_metrics["negative_action_words"],
                "full_all_negative_action_words": full_resume_metrics["all_negative_action_words"],
            }

            analysis_results.append(result_doc)

    return analysis_results


# -------------------------
# RUN SCRIPT
# -------------------------
if __name__ == "__main__":

    analysis_results = compare_action_words()

    df = pd.DataFrame(analysis_results)

    # Convert lists → readable strings
    for col in df.columns:
        df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    # Save Excel
    df.to_csv("action_words_comparison.csv", index=False)

    print("\n✅ Done! File saved as action_words_comparison.csv")