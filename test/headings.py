"""

Test script to compare headings extracted using Original Resuscan code vs Groq (llama-3.1-8b)

"""

from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, fetch_pdf_from_s3, extract_full_text
from extraction.section_mapper import map_content_to_standard_header
from extraction.body_content import filter_body_content
from extraction.section_builder import SectionBuilder 
from evaluation.score import calculate_resume_score
from extraction.pointers import *
from extraction.headings import get_headings
from parsers.yolo_parser import LayoutParser
from modules.personal_details import *
from modules.competencies import *
from modules.presentation import *
from modules.information import *
from utils.ngrams import frequent_dynamic_ngrams, better_frequent_ngrams
from utils.headings import get_headings
from utils.names import *
from utils.bold import get_bold
from utils.document import *
from utils.pronouns import *
from groq_utils.prompts import *
from config.settings import *
import pandas as pd
import json
import os
# from groq_utils.resuscan_groq import filter_body_content, filter_headers_with_groq

DATA_DIR = "documents"

# Pass PDF bytes to YOLO layout parser and get detected blocks
def process_resume(pdf_bytes):

    # STEP 1: Layout parsing
    parser = LayoutParser()
    blocks = parser.parse(pdf_bytes)

    # 👉 Now you can pass this anywhere
    # print("Detected blocks:", blocks[:5])

    # STEP 2: Groq (your existing flow)
    # linewise_content, _, _, _ = filter_body_content(pdf_bytes)

    # headers = filter_headers_with_groq(
    #     linewise_content,
    #     prompt="your_prompt"
    # )

    return {
        "blocks": blocks,
        # "headers": headers
    }

def compare_headings_from_groq_and_yolo():

    analysis_results = []

    for file_name in os.listdir(DATA_DIR):

        if file_name.endswith(".pdf"):
            
            print(f"\n📄 Processing file: {file_name}")

            pdf_path = os.path.join(DATA_DIR, file_name)
            pdf_bytes = load_local_pdf(pdf_path)

            experience = 3
            # file_name = "Ujjwal Tyagi.pdf"

            # pdf text  
            # YOLO Pass
            results = process_resume(pdf_bytes)
            # print(results)

            # Extract Headings using Resuscan Code
            headings, subHeadings, notRequired_Heading, Work_Project_Headings, EduSkill_Headings, Other_Headings, Other_headings_db, sectionMap, sectionMapCount, standard_match_headings, standard_match_headings_count = get_headings(pdf_bytes)
            # printing all values separately:
            print("🔍 get_headings() outputs: ")
            print(f"\nHeadings: {headings}, \nSub Headings: {subHeadings}, \nNot Required heading: {notRequired_Heading}, \nWork Project Headings: {Work_Project_Headings}\n\n")
            # printing remaining headings outputs separately
            print(f"EduSkill_Headings: {EduSkill_Headings}, \nOther_Headings: {Other_Headings}, \nOther_headings_db: {Other_headings_db}, \nsectionMap: {sectionMap}, \nsectionMapCount: {sectionMapCount}, \nstandard_match_headings: {standard_match_headings}, \nstandard_match_headings_count: {standard_match_headings_count}\n\n")
            actualHeadingsCount = len(subHeadings)
            NRlength = len(notRequired_Heading)
            ORlength = len(Other_Headings)
            ORlength_db=len(Other_headings_db)
            

            # Filter Body Content
            linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_bytes)

            # Filter Headers with Groq 
            cleaned_headers = filter_headers_with_groq(linewise_content_with_fonts, cleaned_headers_prompt_template)
            print(f"✅ Cleaned Headers: {cleaned_headers}")

            # Get Standard Headings Map
            standard_headings_map = get_standard_headings_map(cleaned_headers, standard_headings_prompt)
            print(f"✅ Standard Headings Map: {standard_headings_map}")

            # Section Building

            # ensure list type
            import ast
            if isinstance(cleaned_headers, str):
                cleaned_headers = ast.literal_eval(cleaned_headers)

            # -------------------------
            # BUILD SECTIONS
            # -------------------------
            builder = SectionBuilder(cleaned_headers)
            sections = builder.build(results["blocks"])

            print(f"📦 Sections: {list(sections.keys())}")
            with open(SECTIONS_OUTPUT_FILE_PATH, "w") as f:
                json.dump(sections, f, indent=4)

            print(f"💾 Section building complete. Sections saved to {SECTIONS_OUTPUT_FILE_PATH}")
            print(f"📦 Sections: {list(sections.keys())}")

            standard_sections = map_content_to_standard_header(sections, standard_headings_map)

            with open(STANDARD_SECTIONS_OUTPUT_FILE_PATH, "w") as f:
                json.dump(standard_sections, f, indent=4)
            # # YOLO Pass

            result_doc = {
                "file_name": file_name,
                "headings": headings,
                "subHeadings": subHeadings,
                "notRequired_Heading": notRequired_Heading,
                "standard_match_headings": standard_match_headings,
                "cleaned_headers": cleaned_headers,
                "standard_headings_map": standard_headings_map,
            }

            analysis_results.append(result_doc)

    return analysis_results

if __name__ == "__main__":

    analysis_results = compare_headings_from_groq_and_yolo()
    pd.DataFrame(analysis_results).to_csv("headings_comparison_results.csv", index=False)