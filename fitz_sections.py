from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, fetch_pdf_from_s3, extract_full_text
from extraction.section_mapper import map_content_to_standard_header, save_useless_bullets, get_mapped_section_text
# from extraction.body_content import filter_body_content
from extraction.body_content import filter_body_content_new as filter_body_content # also returns word sizes and positions
from extraction.section_builder import SectionBuilder
from extraction.model_inference import *
from evaluation.score import calculate_resume_score
from fitz_mapper import *
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
import time
import json
import os
# from groq_utils.resuscan_groq import filter_body_content, filter_headers_with_groq

MODEL_THRESHOLD = 0.9
DATA_DIR = "documents"

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
def parse_resume_with_yolo(pdf_bytes):

    # STEP 1: Layout parsing
    parser = LayoutParser()
    blocks = parser.parse(pdf_bytes)

    return {
        "blocks": blocks,
    }

def process_resume(pdf_bytes, experience, file_name):

    # YOLO Pass
    results = parse_resume_with_yolo(pdf_bytes)
    # print(results)
    with open("yolo_blocks.json", "w") as f:
        json.dump(results, f, indent=2)

    # Extract Headings using Resuscan Code
    headings, subHeadings, notRequired_Heading, Work_Project_Headings, EduSkill_Headings, Other_Headings, Other_headings_db, sectionMap, sectionMapCount, standard_match_headings, standard_match_headings_count = get_headings(pdf_bytes)
    # printing all values separately:
    print("🔍 get_headings() outputs: ")
    print(f"\nHeadings: {headings}, \nSub Headings: {subHeadings}, \nNot Required heading: {notRequired_Heading}, \nWork Project Headings: {Work_Project_Headings}\n\n")
    actualHeadingsCount = len(subHeadings)
    NRlength = len(notRequired_Heading)
    ORlength = len(Other_Headings)
    ORlength_db=len(Other_headings_db)
    

    # Filter Body Content
    linewise_content_with_fonts, font_and_words, max_font_size, max_words, word_positions = filter_body_content(pdf_bytes)
    # print(f"✅ Body Font Size: {max_font_size}, Words: {max_words}")
    # print(f"✅ Font and Words: {font_and_words}")
    # print(f"✅ Linewise content with fonts: {linewise_content_with_fonts[:5]}")

    # Map YOLO headers with fonts
    yolo_headers_with_fonts = map_yolo_headers_with_fonts(results["blocks"], word_positions)

    # Filter Headers with Groq 
    # cleaned_headers = filter_headers_with_groq(linewise_content_with_fonts, cleaned_headers_prompt_template)
    cleaned_headers = filter_headers_with_groq(yolo_headers_with_fonts, cleaned_headers_prompt_template_v3)
    print(f"✅ Cleaned Headers: {cleaned_headers}")

    time.sleep(5)
    print("⏳ Getting Standard Headings Map...")
    # Get Standard Headings Map from Groq
    # standard_headings_map = get_standard_headings_map(cleaned_headers, standard_headings_prompt)
    # print(f"✅ Standard Headings Map: {standard_headings_map}")
    
    standard_headings_map_groq = get_standard_headings_map(cleaned_headers, standard_headings_prompt_v2)
    print(f"✅ Standard Headings Map Groq: {standard_headings_map_groq}")
    
    # Get Standard headings Map from Valhalla DistilBART
    valhalla_standard_headers = build_map_with_model(cleaned_headers, distilbart_classifier, MODEL_THRESHOLD)

    standard_headings_map = flatten_meta_map(valhalla_standard_headers)

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

    # experience_and_projects_content = standard_sections.get("Experience", "") + " " + standard_sections.get("Projects", "")
    # print(f"✅ Experience and Projects Content: {experience_and_projects_content}")

    # ✅ Extract Experience + Projects using .get("text") and .get("bullets")
    exp_data = standard_sections.get("Experience", {})
    print(f"✅ Experience: {exp_data}")
    proj_data = standard_sections.get("Projects", {})
    print(f"✅ Projects: {proj_data}")

    experience_and_projects_content = (
        exp_data.get("text", "") + " " + proj_data.get("text", "")
    ).strip()

    # mapper = PDFSectionMapper(cleaned_headers)
    # sections = mapper.build_sections(pdf_file_path)

    # for sec, content in sections.items():
    #     print(f"\n=== {sec} ===")
    #     print("\n".join(content[:5]))  # preview

def process_multiple_resume():

    analysis_results = []

    for file_name in os.listdir(DATA_DIR):

        if file_name.endswith(".pdf"):

            start_time = time.time()

            print(f"🔍 Processing resume: {file_name}")
            
            pdf_path = os.path.join(DATA_DIR, file_name)
            pdf_bytes = load_local_pdf(pdf_path)            
            result_doc = process_resume(pdf_bytes, experience=3, file_name=file_name)
            analysis_results.append(result_doc)

            print(f"📊 Resume processed: {file_name}")
            print(f"📊 Time taken: {time.time() - start_time:.2f} seconds")


    return analysis_results

if __name__ == "__main__":

    # For local testing    
    # SINGLE RESUME
    # convert local PDF to bytes
    # file_name = "Aagam_Shah_Resume.pdf"
    file_name = "Vinay_P_12042026 (1).pdf"
    pdf_path = f"{DATA_DIR}/{file_name}"
    pdf_bytes = load_local_pdf(pdf_path)
    
    # For S3 PDF
    # pdf_bytes, pdf_file2 = fetch_pdf_from_s3(
    #     "https://local-job-match-pro.s3.ap-south-2.amazonaws.com/e9168491d6ec8e5c0fcdaced9072de5b",
    #     os.getenv("AWS_ACCESS_KEY"),  # "your_aws_access_key"
    #     os.getenv("AWS_SECRET_KEY")   #
    # )
    
    experience = 3
    file_name = "Ujjwal Tyagi.pdf"

    output = process_resume(pdf_bytes, experience, file_name)


    # BULK RESUME

    # start_time = time.time()
    # analysis_results = process_multiple_resume()

    # pd.DataFrame(analysis_results).to_csv("new_prompt_model_metrics_2704.csv", index=False)
    # print(f"📊 Total time taken: {time.time() - start_time:.2f} seconds")
