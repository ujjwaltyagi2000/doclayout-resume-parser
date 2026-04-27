"""

Test script to compare headings extracted using Original Resuscan code vs Groq (llama-3.1-8b)

"""

from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, fetch_pdf_from_s3, extract_full_text
from extraction.section_mapper import map_content_to_standard_header
# from extraction.body_content import filter_body_content
from extraction.body_content import filter_body_content_new as filter_body_content
from test.model_inference import *
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
import time
import os
# from groq_utils.resuscan_groq import filter_body_content, filter_headers_with_groq

DATA_DIR = "documents"

threshold = 0.9 # for model inference

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

    # STEP 1: Layout parsing
    parser = LayoutParser()
    blocks = parser.parse(pdf_bytes)

    return {
        "blocks": blocks,
        # "headers": headers
    }

def single_resume_processing(file_name):

    print(f"\n📄 Processing file: {file_name}")

    pdf_path = os.path.join(DATA_DIR, file_name)
    pdf_bytes = load_local_pdf(pdf_path)
    
    results = process_resume(pdf_bytes)
    print(results)
        
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
    linewise_content_with_fonts, font_and_words, max_font_size, max_words, word_positions = filter_body_content(pdf_bytes)

    # Map YOLO headers with fonts
    yolo_headers_with_fonts = map_yolo_headers_with_fonts(results["blocks"], word_positions)


    # Filter Headers with Groq 
    cleaned_headers = filter_headers_with_groq(yolo_headers_with_fonts, cleaned_headers_prompt_template_v3)
    print(f"✅ Cleaned Headers: {cleaned_headers}")

    time.sleep(5)
    print("⏳ Getting Standard Headings Map...")
    # Get Standard Headings Map
    standard_headings_map = get_standard_headings_map(cleaned_headers, standard_headings_prompt)
    print(f"✅ Standard Headings Map: {standard_headings_map}")
    time.sleep(5)
    standard_headings_map_v2 = get_standard_headings_map(cleaned_headers, standard_headings_prompt_v2)
    print(f"✅ Standard Headings Map V2: {standard_headings_map_v2}")

    # meta_standard_headers = build_meta_standard_headers(cleaned_headers)
    meta_standard_headers = build_map_with_model(cleaned_headers, zero_shot_classifier, threshold)
    print(f"✅ Meta Standard Headers: {meta_standard_headers}")
    valhalla_standard_headers = build_map_with_model(cleaned_headers, distilbart_classifier, threshold)
    print(f"✅ Valhalla Standard Headers: {valhalla_standard_headers}")

    result_doc = {
        "file_name": file_name,
        "headings": headings,
        "subHeadings": subHeadings,
        "notRequired_Heading": notRequired_Heading,
        "standard_match_headings": standard_match_headings,
        "cleaned_headers": cleaned_headers,
        "standard_headings_map": standard_headings_map,
        "standard_headings_map_v2": standard_headings_map_v2,
        "meta_standard_headers": meta_standard_headers,
        "valhalla_standard_headers": valhalla_standard_headers
    }

    return result_doc


def compare_headings_from_groq_and_yolo():

    analysis_results = []

    for file_name in os.listdir(DATA_DIR):

        if file_name.endswith(".pdf"):
            
            time.sleep(5)
            
            result_doc = single_resume_processing(file_name)
            analysis_results.append(result_doc)


    return analysis_results

if __name__ == "__main__":

    # process a single resume
    # file_name = "Ajay Data Analyst.pdf"
    # result_doc = single_resume_processing(file_name)

    # print(result_doc)

    # process multiple resumes
    analysis_results = compare_headings_from_groq_and_yolo()
    pd.DataFrame(analysis_results).to_csv("headings_comparison_groq_zeroshot_2404.csv", index=False)