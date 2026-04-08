from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, fetch_pdf_from_s3, extract_full_text
from extraction.section_mapper import map_content_to_standard_header
from extraction.body_content import filter_body_content
from extraction.section_builder import SectionBuilder 
from extraction.headings import get_headings
from parsers.yolo_parser import LayoutParser
from modules.personal_details import *
from modules.competencies import *
from modules.presentation import *
from modules.information import *
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
    pdf_path = "Puunita Chaturvedi.pdf"
    pdf_bytes = load_local_pdf(pdf_path)
    
    # For S3 PDF
    pdf_bytes = fetch_pdf_from_s3(
        "https://local-job-match-pro.s3.ap-south-2.amazonaws.com/e9168491d6ec8e5c0fcdaced9072de5b",
        os.getenv("AWS_ACCESS_KEY"),  # "your_aws_access_key"
        os.getenv("AWS_SECRET_KEY")   #
    )

    # pdf text

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
    with open(SECTIONS_OUTPUT_FILE_PATH, "w") as f:
        json.dump(sections, f, indent=4)

    print(f"💾 Section building complete. Sections saved to {SECTIONS_OUTPUT_FILE_PATH}")
    print(f"📦 Sections: {list(sections.keys())}")

    standard_sections = map_content_to_standard_header(sections, standard_headings_map)

    with open(STANDARD_SECTIONS_OUTPUT_FILE_PATH, "w") as f:
        json.dump(standard_sections, f, indent=4)

    experience_and_projects_content = standard_sections.get("Experience", "") + " " + standard_sections.get("Projects", "")
    # print(f"✅ Experience and Projects Content: {experience_and_projects_content}")

    # INFORMATION MENU METRICS
    action_words, total_action_words, all_action_words = get_action_words(experience_and_projects_content)

    print("Action Words:", action_words)
    print("Total Action Words:", total_action_words)
    print("All Action Words:", all_action_words)

    frequency_list, total_frequent_action_words, repeated_frequency = frequency_Action_words(all_action_words)
    print("Frequent Action Words (appearing at least 3 times):", frequency_list)
    print("Total Frequent Action Words:", total_frequent_action_words)
    print("Repeated Frequency of Action Words:", repeated_frequency)

    negative_action_words, total_negative_action_words, all_negative_action_words = get_negative_action_words(experience_and_projects_content)
    print("Negative Action Words:", negative_action_words)
    print("Total Negative Action Words:", total_negative_action_words)
    print("All Negative Action Words:", all_negative_action_words)

    filler_words, total_filler_words, all_filler_words = get_filler_words(experience_and_projects_content)
    print("Filler Words:", filler_words)
    print("Total Filler Words:", total_filler_words)
    print("All Filler Words:", all_filler_words)

    voice = text_voice(experience_and_projects_content)
    print("Passive Voice Constructions:", voice)


    # PRESENTATION MENU METRICS
    font_styles,standard_font_style_flag,multiple_font_style,font_sizes,multiple_font_size = get_font_style_size(pdf_bytes)
    print("Font Styles:", font_styles)
    print("Standard Font Style Flag:", standard_font_style_flag)
    print("Multiple Font Styles:", multiple_font_style)
    print("Font Sizes:", font_sizes)
    print("Multiple Font Sizes:", multiple_font_size)

    resume_text = extract_full_text(pdf_bytes)
    print(f"✅ Extracted Resume Text (first 50 chars): {resume_text[:50]}")

    # PERSONAL DETAILS MENU METRICS
    phones, phones1, phones2, all_phones = get_phones(resume_text)
    print("Phone Numbers (E164 format):", phones)
    print("Phone Numbers (Regex 1):", phones1)
    print("Phone Numbers (Regex 2):", phones2)
    print("All Phone Numbers:", all_phones)

    reg_Phone = get_Phones(resume_text)
    print("Phone Numbers (Regex 3):", reg_Phone)

    email_finderSet =  get_emails(resume_text) 
    print("Email Addresses:", email_finderSet)

    url,linkedIn_flag,url_flag =  get_url(pdf_bytes)
    print("URLs:", url)
    print("LinkedIn Flag:", linkedIn_flag)
    print("URL Flag:", url_flag)

    images = check_Images(pdf_bytes)
    print("Images Found:", images)

    # COMPETENCIES MENU METRICS

    finalBullet,Bullets_Total,standard_bullet_flag= get_bullets(resume_text)
    print("Bullets Found:", finalBullet)
    print("Total Bullets:", Bullets_Total)
    print("Standard Bullet Flag:", standard_bullet_flag)

    ats_date =  getATS_dates(resume_text)
    print("ATS Dates Found:", ats_date)
    dates_nonAts =  get_nonATSdates(resume_text)
    print("Non ATS Dates Found:", dates_nonAts)
    dates = ats_date + dates_nonAts

    measurable= get_namedEntityMeasurable(resume_text)
    print("Measurable Named Entities:", measurable)
    clean_measurable =  get_measurableUpdated(resume_text,finalBullet,measurable,all_phones,dates)
    print("Clean Measurable Strings:", clean_measurable)

    # save outputs to a file
    output_data = {
        "action_words": action_words,
        "total_action_words": total_action_words,
        "all_action_words": all_action_words,
        "frequent_action_words": frequency_list,
        "total_frequent_action_words": total_frequent_action_words,
        "repeated_frequency": repeated_frequency,
        "negative_action_words": negative_action_words,
        "total_negative_action_words": total_negative_action_words,
        "all_negative_action_words": all_negative_action_words,
        "filler_words": filler_words,
        "total_filler_words": total_filler_words,
        "all_filler_words": all_filler_words,
        "passive_voice_constructions": voice,
        "font_styles": font_styles,
        "standard_font_style_flag": standard_font_style_flag,
        "multiple_font_style": multiple_font_style,
        "font_sizes": font_sizes,
        "multiple_font_size": multiple_font_size,
        "phone_numbers": all_phones,
        "reg_Phone": reg_Phone,
        "email_finderSet": list(email_finderSet),
        "url": url,
        "linkedIn_flag": linkedIn_flag,
        "url_flag": url_flag,
        "images_found": images,
        "finalBullet": finalBullet,
        "Bullets_Total": Bullets_Total,
        "standard_bullet_flag": standard_bullet_flag,
        "ats_date": ats_date,
        "dates_nonAts": dates_nonAts,
        "measurable_ner": measurable,
        "clean_measurable": clean_measurable
    }

    with open(FINAL_OUTPUT_FILE_PATH, "w") as f:
        json.dump(output_data, f, indent=4)