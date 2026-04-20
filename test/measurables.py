from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, extract_full_text
from extraction.section_mapper import map_content_to_standard_header
from extraction.body_content import filter_body_content
from extraction.section_builder import SectionBuilder 
from extraction.headings import get_headings
from parsers.yolo_parser import LayoutParser
from groq_utils.prompts import *


from modules.personal_details import *
from modules.competencies import *
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
# STEP 2: Measurable Metrics Function
# -------------------------
def compute_measurable_metrics(text, bullets, phones, dates):

    measurable = get_namedEntityMeasurable(text)

    clean_measurable = get_measurableUpdated(
        text,
        bullets,
        measurable,
        phones,
        dates
    )

    return {
        "raw_measurable": measurable,
        "clean_measurable": clean_measurable,
        "total_raw_measurable": len(measurable) if measurable else 0,
        "total_clean_measurable": len(clean_measurable) if clean_measurable else 0
    }


# -------------------------
# MAIN FUNCTION
# -------------------------
def compare_measurables():

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
            # STEP 2: Headings
            # -------------------------
            headings, subHeadings, notRequired_Heading, Work_Project_Headings, EduSkill_Headings, Other_Headings, Other_headings_db, sectionMap, sectionMapCount, standard_match_headings, standard_match_headings_count = get_headings(pdf_bytes)

            # -------------------------
            # STEP 3: Body Content
            # -------------------------
            linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_bytes)

            # -------------------------
            # STEP 4: Clean Headers
            # -------------------------
            cleaned_headers = filter_headers_with_groq(linewise_content_with_fonts, cleaned_headers_prompt_template)

            if isinstance(cleaned_headers, str):
                cleaned_headers = ast.literal_eval(cleaned_headers)

            # -------------------------
            # STEP 5: Standard Mapping
            # -------------------------
            standard_headings_map = get_standard_headings_map(cleaned_headers, standard_headings_prompt)

            # -------------------------
            # STEP 6: Section Building
            # -------------------------
            builder = SectionBuilder(cleaned_headers)
            sections = builder.build(results["blocks"])

            standard_sections = map_content_to_standard_header(sections, standard_headings_map)

            # -------------------------
            # Extract Resume Text
            # -------------------------
            full_resume_text = extract_full_text(pdf_bytes)

            # -------------------------
            # Common Data Needed
            # -------------------------
            finalBullet, Bullets_Total, standard_bullet_flag = get_bullets(full_resume_text)

            ats_date = getATS_dates(full_resume_text)
            dates_nonAts = get_nonATSdates(full_resume_text)
            dates = ats_date + dates_nonAts

            phones, _, _, all_phones = get_phones(full_resume_text)

            # -------------------------
            # CASE 1: Experience + Projects
            # -------------------------
            exp_data = standard_sections.get("Experience", {})
            proj_data = standard_sections.get("Projects", {})

            exp_proj_text = (
                exp_data.get("text", "") + " " + proj_data.get("text", "")
            ).strip()

            exp_proj_bullets = exp_data.get("bullets", []) + proj_data.get("bullets", [])

            exp_proj_metrics = compute_measurable_metrics(
                exp_proj_text,
                exp_proj_bullets,
                all_phones,
                dates
            )

            # -------------------------
            # CASE 2: Full Resume
            # -------------------------
            full_metrics = compute_measurable_metrics(
                full_resume_text,
                finalBullet,
                all_phones,
                dates
            )

            # -------------------------
            # RESULT DOC
            # -------------------------
            result_doc = {
                "file_name": file_name,

                # -------- EXP + PROJECT --------
                "exp_proj_raw_measurable": exp_proj_metrics["raw_measurable"],
                "exp_proj_clean_measurable": exp_proj_metrics["clean_measurable"],
                "exp_proj_total_raw": exp_proj_metrics["total_raw_measurable"],
                "exp_proj_total_clean": exp_proj_metrics["total_clean_measurable"],

                # -------- FULL RESUME --------
                "full_raw_measurable": full_metrics["raw_measurable"],
                "full_clean_measurable": full_metrics["clean_measurable"],
                "full_total_raw": full_metrics["total_raw_measurable"],
                "full_total_clean": full_metrics["total_clean_measurable"],
            }

            analysis_results.append(result_doc)

    return analysis_results


# -------------------------
# RUN SCRIPT
# -------------------------
if __name__ == "__main__":

    analysis_results = compare_measurables()

    df = pd.DataFrame(analysis_results)

    # Make lists readable
    for col in df.columns:
        df[col] = df[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)

    df.to_csv("measurable_comparison.csv", index=False)

    print("\n✅ Done! File saved as measurable_comparison.csv")