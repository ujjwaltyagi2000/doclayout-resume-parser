from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, fetch_pdf_from_s3
from extraction.body_content import filter_body_content
from extraction.section_builder import SectionBuilder 
from extraction.headings import get_headings
from parsers.yolo_parser import LayoutParser
from groq_utils.prompts import *
from config.settings import *
import json
import os
# from groq_utils.resuscan_groq import filter_body_content, filter_headers_with_groq

# Pass PDF bytes to YOLO layout parser and get detected blocks
def process_resume(pdf_bytes):

    # STEP 1: Layout parsing
    parser = LayoutParser()
    blocks = parser.parse(pdf_bytes)

    # 👉 Now you can pass this anywhere
    print("Detected blocks:", blocks[:5])

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

if __name__ == "__main__":

    # For local testing
    
    # convert local PDF to bytes
    # pdf_path = "Puunita Chaturvedi.pdf"
    # pdf_bytes = load_local_pdf(pdf_path)
    
    # For S3 PDF
    pdf_bytes = fetch_pdf_from_s3(
        "https://local-job-match-pro.s3.ap-south-2.amazonaws.com/e9168491d6ec8e5c0fcdaced9072de5b",
        os.getenv("AWS_ACCESS_KEY"),  # "your_aws_access_key"
        os.getenv("AWS_SECRET_KEY")   #
    )

    # YOLO Pass
    results = process_resume(pdf_bytes)
    print(results)

    # Extract Headings using Resuscan Code
    # headings, subHeadings, notRequired_Heading, Work_Project_Headings, EduSkill_Headings, Other_Headings, Other_headings_db, sectionMap, sectionMapCount, standard_match_headings, standard_match_headings_count = get_headings(pdf_bytes)
    # # printing all values separately:
    # print("🔍 get_headings() outputs: ")
    # print(f"\nHeadings: {headings}, \nSub Headings: {subHeadings}, \nNot Required heading: {notRequired_Heading}, \nWork Project Headings: {Work_Project_Headings}\n\n")
    # actualHeadingsCount = len(subHeadings)
    # NRlength = len(notRequired_Heading)
    # ORlength = len(Other_Headings)
    # ORlength_db=len(Other_headings_db)
    

    # Filter Body Content
    linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_bytes)
    print(f"✅ Body Font Size: {max_font_size}, Words: {max_words}")
    print(f"✅ Font and Words: {font_and_words}")
    print(f"✅ Linewise content with fonts: {linewise_content_with_fonts[:5]}")

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
    with open(SECTIONS_OUPUT_FILE_PATH, "w") as f:
        json.dump(sections, f, indent=4)

    print(f"💾 Section building complete. Sections saved to {SECTIONS_OUPUT_FILE_PATH}")
    # print(f"📦 Sections: {list(sections.keys())}")