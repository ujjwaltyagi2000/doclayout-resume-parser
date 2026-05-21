from groq_utils.cleaned_headers import filter_headers_with_groq, get_standard_headings_map
from parsers.pdf_parser import load_local_pdf, fetch_pdf_from_s3, extract_full_text
from extraction.section_mapper import map_content_to_standard_header, save_useless_bullets, get_mapped_section_text
# from extraction.body_content import filter_body_content
from extraction.body_content import filter_body_content_new as filter_body_content # also returns word sizes and positions
from extraction.section_builder import SectionBuilder
from extraction.model_inference import *
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
from check_multi_colsv2 import detect_columns_advanced
import pandas as pd
import ahocorasick
import time
import json
import re
import os
# from groq_utils.resuscan_groq import filter_body_content, filter_headers_with_groq

MODEL_THRESHOLD = 0.9
BATCH_SIZE = 10
# DATA_DIR = "documents"
DATA_DIR = "Check PDFs"
OUTPUT_DIR = "sections_output_2105"

SECTIONS_DIR = f"{OUTPUT_DIR}/sections"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# OUTPUT_DIR = "output2"

# -----------------------------------
# Normalize helper
# -----------------------------------
def normalize_text(text):
    text = str(text).strip().lower()

    # replace &, / etc with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -----------------------------------
# Build Automaton
# -----------------------------------
def build_header_automaton(file_path):

    df = pd.read_excel(file_path)

    # second column
    keywords = df.iloc[:, 1].dropna().astype(str).tolist()

    A = ahocorasick.Automaton()

    for keyword in keywords:

        normalized_keyword = normalize_text(keyword)

        if normalized_keyword:
            A.add_word(normalized_keyword, keyword)

    A.make_automaton()

    return A

HEADER_AUTOMATON = build_header_automaton(
    "https://s3.ap-south-1.amazonaws.com/mployee.me/keywords_list/Headlines.xlsx"
)

# -----------------------------------
# Match headers against automaton
# -----------------------------------
def match_headers_with_automaton(headers, automaton):

    matched_headers = []

    for h in headers:

        text = h["text"]
        normalized_text = normalize_text(text)

        found_match = False

        for _, matched_keyword in automaton.iter(normalized_text):

            # exact normalized match only
            if normalize_text(matched_keyword) == normalized_text:

                matched_headers.append({
                    "text": text,
                    "font": h["font"],
                    "y_position": h["y_position"],
                    "matched_keyword": matched_keyword
                })

                found_match = True
                break

    return matched_headers

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

def process_resume(pdf_bytes, pdf_path, experience, file_name):

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
    print(f"✅ Font and Words: {font_and_words}")
    # print(f"✅ Linewise content with fonts: {linewise_content_with_fonts[:5]}")

    has_same_font_size = False
    same_font_size = None

    if len(font_and_words) == 1:
        has_same_font_size = True
        same_font_size = font_and_words[0]["size"]
        print(f"✅ All words have the same font size: {same_font_size}")

    # Map YOLO headers with fonts
    yolo_headers_with_fonts = map_yolo_headers_with_fonts(results["blocks"], word_positions)
    print(f"✅ YOLO Headers with Fonts: {yolo_headers_with_fonts}")

    # Filter Headers with Groq 
    # cleaned_headers = filter_headers_with_groq(linewise_content_with_fonts, cleaned_headers_prompt_template)
    cleaned_headers, prompt_tokens, completion_tokens, total_tokens = filter_headers_with_groq(yolo_headers_with_fonts, cleaned_headers_prompt_template_v3)
    print(f"✅ Cleaned Headers: {cleaned_headers}")

    if isinstance(cleaned_headers, str):
        import ast
        cleaned_headers = ast.literal_eval(cleaned_headers)
        print(f"Cleaned Headers count: {len(cleaned_headers)}")

    # TEST CODE TO FIND HEADERS THAT ARE WRONGFULLY MISSED OR INCLUDED
    # =============================================================================================================

    from collections import Counter

    # -----------------------------------
    # Normalize cleaned headers
    # -----------------------------------
    cleaned_headers_set = set(h.strip() for h in cleaned_headers)

    # -----------------------------------
    # Get cleaned header objects
    # -----------------------------------
    cleaned_header_objects = []

    for h in yolo_headers_with_fonts:

        if h["text"].strip() in cleaned_headers_set:
            cleaned_header_objects.append(h)

    # -----------------------------------
    # 1. Max recurring font size
    #    ONLY from cleaned headers
    # -----------------------------------
    cleaned_fonts = [
        h["font"]
        for h in cleaned_header_objects
        if h["font"] is not None
    ]

    font_counter = Counter(cleaned_fonts)

    max_recurring_font_size = None

    if font_counter:
        max_recurring_font_size = font_counter.most_common(1)[0][0]

    # -----------------------------------
    # 2. Headers in YOLO input
    #    having same dominant font
    #    BUT missing from cleaned headers
    # -----------------------------------
    max_font_missing_headers = []

    if max_recurring_font_size is not None:

        for h in yolo_headers_with_fonts:

            if h["font"] == max_recurring_font_size:

                if h["text"].strip() not in cleaned_headers_set:

                    max_font_missing_headers.append(h)

    # -----------------------------------
    # 3. Headers PRESENT in cleaned headers
    #    BUT font size NOT same (+/-1 excluded)
    # -----------------------------------
    different_font_cleaned_headers = []

    if max_recurring_font_size is not None:

        for h in cleaned_header_objects:

            if h["font"] is None:
                continue

            # NOT within +/-1
            if abs(h["font"] - max_recurring_font_size) > 1:

                different_font_cleaned_headers.append(h)

    # -----------------------------------
    # 4. Split cleaned headers into:
    #    A) dominant font size headers
    #    B) different font size headers
    # -----------------------------------

    max_font_size_headers = []
    different_font_size_headers = []

    if max_recurring_font_size is not None:

        for h in cleaned_header_objects:

            if h["font"] is None:
                continue

            # same dominant font
            if h["font"] == max_recurring_font_size:

                max_font_size_headers.append(h)

            else:

                different_font_size_headers.append(h)

    # -----------------------------------
    # Match missing same-font headers
    # against keyword automaton
    # -----------------------------------

    matched_headers_from_groq_input = match_headers_with_automaton(
        max_font_missing_headers,
        HEADER_AUTOMATON
    )

    print("\n✅ Matched Headers From Groq Input:")
    print(matched_headers_from_groq_input)

    # -----------------------------------
    # Debug prints
    # -----------------------------------
    print(f"\n📄 Max font size headers: \n{max_font_size_headers}")
    print(f"\n🔃 Different font size headers: \n{different_font_size_headers}")

    print("\n✅ Max Recurring Font Size (from cleaned headers):")
    print(max_recurring_font_size)

    print("\n❌ Same Font Headers Missing From Cleaned Headers:")
    print(max_font_missing_headers)

    max_font_missing_headers_list = [h["text"] for h in max_font_missing_headers]

    print(f"\n⚠️ Same Font Headers Missing From Cleaned Headers: \n{max_font_missing_headers_list}")

    # adding max_font_missing_headers_list to cleaned_headers
    cleaned_headers.extend(max_font_missing_headers_list)
    print(f"\n✅ Cleaned Headers: {len(cleaned_headers)}")

    print("\n⚠️ Cleaned Headers Having Different Font Size:")
    print(different_font_cleaned_headers)
    # =============================================================================================================


    # time.sleep(5)
    # print("⏳ Getting Standard Headings Map...")
    # Get Standard Headings Map from Groq
    # standard_headings_map = get_standard_headings_map(cleaned_headers, standard_headings_prompt)
    # print(f"✅ Standard Headings Map: {standard_headings_map}")
    
    # standard_headings_map_groq = get_standard_headings_map(cleaned_headers, standard_headings_prompt_v2)
    # print(f"✅ Standard Headings Map Groq: {standard_headings_map_groq}")
    
    # Get Standard headings Map from Valhalla DistilBART
    # valhalla_standard_headers = build_map_with_model(cleaned_headers, distilbart_classifier, MODEL_THRESHOLD)

    # print(f"\n ✅ Valhalla Standard Headings Map: {valhalla_standard_headers}")

    # standard_headings_map = flatten_meta_map(valhalla_standard_headers)

    # print(f"✅ Standard Headings Map: {standard_headings_map}")

    # Section Building

    # ensure list type
    import ast
    if isinstance(cleaned_headers, str):
        cleaned_headers = ast.literal_eval(cleaned_headers)

    # # -------------------------
    # # BUILD SECTIONS
    # # -------------------------

    # detect if resume is single column or multi column
    is_multi_column = detect_columns_advanced(pdf_path, max_pages=5).is_multicolumn
    print("🔍 Is Multi Column: ", is_multi_column)

    # build sections
    builder = SectionBuilder(cleaned_headers)
    sections = builder.build(results["blocks"], is_multi_column)

    print(f"📦 Sections: {list(sections.keys())}")

    clean_file_name = file_name.removesuffix(".pdf")

    with open(f"{SECTIONS_DIR}/{clean_file_name}.json", "w") as f:
        json.dump(sections, f, indent=4)
    # with open(SECTIONS_OUTPUT_FILE_PATH, "w") as f:
    #     json.dump(sections, f, indent=4)

    print(f"💾 Section building complete. Sections saved to {SECTIONS_OUTPUT_FILE_PATH}")
    print(f"📦 Sections: {list(sections.keys())}")

    # standard_sections = map_content_to_standard_header(sections, standard_headings_map)

    # # with open(STANDARD_SECTIONS_OUTPUT_FILE_PATH, "w") as f:
    # #     json.dump(standard_sections, f, indent=4)

    # # experience_and_projects_content = standard_sections.get("Experience", "") + " " + standard_sections.get("Projects", "")
    # # print(f"✅ Experience and Projects Content: {experience_and_projects_content}")

    # # ✅ Extract Experience + Projects using .get("text") and .get("bullets")
    # exp_data = standard_sections.get("Experience", {})
    # # print(f"✅ Experience: {exp_data}")
    # proj_data = standard_sections.get("Projects", {})
    # # print(f"✅ Projects: {proj_data}")

    # experience_and_projects_content = (
    #     exp_data.get("text", "") + " " + proj_data.get("text", "")
    # ).strip()

    # # combined_list = exp_data.get("bullets", []) + proj_data.get("bullets", [])

    # # bullet_analysis = save_useless_bullets(combined_list)

    # # print("Useless Bullets:", bullet_analysis["useless_bullets"])
    # # print("Total Useless:", bullet_analysis["total_useless"])

    # # first_words = extract_first_words(combined_list)
    # # print(f"🚀 First words extracted from bullets: \n{first_words}")
    # # # print(first_words)

    # # action_words_result = analyze_first_words(first_words)
    # # print(f"\n📃 Action Words Analysis Result: \n{action_words_result}")
    # # print(action_words_result)

    # # # ✅ Save to files
    # # with open("experience_projects_text.txt", "w") as f:
    # #     f.write(experience_and_projects_content)

    # # with open("experience_projects_bullets.txt", "w") as f:
    # #     for index, bullet in enumerate(combined_list):
    # #         f.write(f"[{index + 1}] {bullet}\n")

    # # INFORMATION MENU METRICS
    # # action_words, total_action_words, all_action_words = get_action_words(resume_text) # full resume text
    # print(f"\n📄 Experience and Projects Content: \n{experience_and_projects_content}")
    # action_words, total_action_words, all_action_words = get_action_words(experience_and_projects_content) # experience and projects content only

    # print("Action Words:", action_words)
    # print("Total Action Words:", total_action_words)
    # print("All Action Words:", all_action_words)

    # frequency_list, total_frequent_action_words, repeated_frequency = frequency_Action_words(all_action_words)
    # print("Frequent Action Words (appearing at least 3 times):", frequency_list)
    # print("Total Frequent Action Words:", total_frequent_action_words)
    # print("Repeated Frequency of Action Words:", repeated_frequency)

    # # negative_action_words, total_negative_action_words, all_negative_action_words = get_negative_action_words(resume_text) # full resume text
    # negative_action_words, total_negative_action_words, all_negative_action_words = get_negative_action_words(experience_and_projects_content) # experience and projects content only
    # print("Negative Action Words:", negative_action_words)
    # print("Total Negative Action Words:", total_negative_action_words)
    # print("All Negative Action Words:", all_negative_action_words)

    # frequencyList_negative,total_repeated_actionwords_negative,repeated_frequency_negative =  frequency_Action_words(all_negative_action_words)
    # print("Negative Action Words (appearing at least 3 times):", frequencyList_negative)
    # print("Total Negative Action Words:", total_repeated_actionwords_negative)
    # print("Repeated Frequency of Negative Action Words:", repeated_frequency_negative)

    # filler_words, total_filler_words, all_filler_words = get_filler_words(experience_and_projects_content)
    # print("Filler Words:", filler_words)
    # print("Total Filler Words:", total_filler_words)
    # print("All Filler Words:", all_filler_words)

    # voice = text_voice(experience_and_projects_content)
    # print("Passive Voice Constructions:", voice)


    # # PRESENTATION MENU METRICS
    # font_styles,standard_font_style_flag,multiple_font_style,font_sizes,multiple_font_size = get_font_style_size(pdf_bytes)
    # print("Font Styles:", font_styles)
    # print("Standard Font Style Flag:", standard_font_style_flag)
    # print("Multiple Font Styles:", multiple_font_style)
    # print("Font Sizes:", font_sizes)
    # print("Multiple Font Sizes:", multiple_font_size)

    # resume_text = extract_full_text(pdf_bytes)
    # print(f"✅ Extracted Resume Text (first 50 chars): {resume_text[:50]}")

    # # PERSONAL DETAILS MENU METRICS
    # phones, phones1, phones2, all_phones = get_phones(resume_text)
    # print("Phone Numbers (E164 format):", phones)
    # print("Phone Numbers (Regex 1):", phones1)
    # print("Phone Numbers (Regex 2):", phones2)
    # print("All Phone Numbers:", all_phones)

    # reg_Phone = get_Phones(resume_text)
    # print("Phone Numbers (Regex 3):", reg_Phone)

    # email_finderSet =  get_emails(resume_text) 
    # print("Email Addresses:", email_finderSet)

    # url,linkedIn_flag,url_flag =  get_url(pdf_bytes)
    # print("URLs:", url)
    # print("LinkedIn Flag:", linkedIn_flag)
    # print("URL Flag:", url_flag)

    # images = check_Images(pdf_bytes)
    # print("Images Found:", images)

    # # COMPETENCIES MENU METRICS

    # finalBullet,Bullets_Total,standard_bullet_flag= get_bullets(resume_text)
    # print("Bullets Found:", finalBullet)
    # print("Total Bullets:", Bullets_Total)
    # print("Standard Bullet Flag:", standard_bullet_flag)

    # ats_date =  getATS_dates(resume_text)
    # print("ATS Dates Found:", ats_date)
    # dates_nonAts =  get_nonATSdates(resume_text)
    # print("Non ATS Dates Found:", dates_nonAts)
    # dates = ats_date + dates_nonAts

    # # measurable= get_namedEntityMeasurable(resume_text) # full resume text
    # measurable = get_namedEntityMeasurable(experience_and_projects_content) # experience and projects content only
    # print("Measurable Named Entities:", measurable)
    # # clean_measurable =  get_measurableUpdated(resume_text,finalBullet,measurable,all_phones,dates) # full resume text
    # clean_measurable = get_measurableUpdated(experience_and_projects_content,finalBullet,measurable,all_phones,dates) # experience and projects content only
    # print("Clean Measurable Strings:", clean_measurable)

    # skills,Skills_Total =  extract_skills(resume_text)
    # print("Extracted Skills:", skills)
    # print("Total Skills Extracted:", Skills_Total)

    # tables=get_tables(pdf_bytes)
    # tables2=get_tables2(pdf_bytes)
    # tables_flag=0
    # if(tables=="table" and tables2 == "table"):
    #     tables_flag=1
    # if(tables=="table"):
    #     tables_flag=1
    # print("Tables Found (Method 1):", tables)
    # print("Tables Found (Method 2):", tables2)
    # print("Tables Flag:", tables_flag)

    # # file_size_kb,file_type,flag_file_size= get_fileDetails(pdf_file2)
    # # print("File Size (KB):", file_size_kb)
    # # print("File Type:", file_type)
    # # print("File Size Flag (<=500KB):", flag_file_size)

    # # MISCELLANEOUS MENU METRICS

    # max_size, content_size_flag = get_maxSize_words(pdf_bytes)
    # print("Max Font Size:", max_size)
    # print("Content Size Flag (1 if max font size is 10, 11, or 12):", content_size_flag)

    # font_colors,font_colors_Total,standard_color_flag= get_font_color(pdf_bytes)
    # print("Font Colors Found:", font_colors)
    # print("Total Font Colors Found:", font_colors_Total)
    # print("Standard Color Flag (1 if only black or dark colors found):", standard_color_flag)

    # total_word_count = get_totalWordCount(resume_text)
    # print("Total Word Count:", total_word_count)

    # pages_count =  get_pageCount(pdf_bytes)
    # print("Total Pages in Resume:", pages_count)

    # personalPronouns = get_excel_pronouns(resume_text)
    # print("Personal Pronouns:", personalPronouns)

    # headings,subHeadings,notRequired_Heading,Work_Project_Headings,EduSkill_Headings,Other_Headings,Other_headings_db,sectionMap,sectionMapCount,standard_match_headings,standard_match_headings_count =  get_headings(pdf_bytes)
    # actualHeadingsCount = len(subHeadings)
    # NRlength = len(notRequired_Heading)
    # ORlength = len(Other_Headings)
    # ORlength_db=len(Other_headings_db)
    # print("Actual Headings Count:", actualHeadingsCount)
    # print("NRlength:", NRlength)
    # print("ORlength:", ORlength)
    # print("ORlength_db:", ORlength_db)

    # # REPEATED WORDS

    # extra_words = ['\uf0d8','\uf0b7',':','/','|']
    # combined_data = ats_date + dates_nonAts + all_phones + finalBullet + list(email_finderSet) + extra_words
    # combined_data = [word for sublist in combined_data for word in sublist.split()]
    # raw_data2 = get_bold(pdf_bytes)
    # repeated_words = frequent_dynamic_ngrams(raw_data2,combined_data)
    # print("Repeated Words (after removing combined data):\n", repeated_words) 

    # # new ngrams function
    # words = raw_data2.split()
    # # cleaned_words = [w for w in words if w not in combined_data]

    # # result = better_frequent_ngrams(raw_data2, combined_data, min_n=4, max_n=20)

    # # print("\n✅ Counter-based Frequent n-grams Result:\n", result)


    # filtered_text, filtered_bullets = get_mapped_section_text(
    #     sections,
    #     standard_headings_map
    # )

    # # Optional: combine bullets too
    # final_text = filtered_text + " " + " ".join(filtered_bullets)

    # repeated_words_new = better_frequent_ngrams(
    #     final_text,
    #     combined_data,
    #     min_n=4,
    #     max_n=20
    # )

    # print("\n✅ Filtered Frequent n-grams Result:\n", repeated_words_new)

    # alloutput = detect_names_all(pdf_bytes,resume_text,extra_words)
    # name_output = get_finalise_names(alloutput)



    # # calculating final score
    # resume_score, score_array = calculate_resume_score(
    #     standard_font_style_flag=standard_font_style_flag,
    #     multiple_font_style=multiple_font_style,
    #     multiple_font_size=multiple_font_size,
    #     content_size_flag=content_size_flag,
    #     font_colors_Total=font_colors_Total,
    #     standard_color_flag=standard_color_flag,
    #     actionwords_total=total_action_words,
    #     actionwords_total_negative=total_negative_action_words,
    #     total_repeated_actionwords_negative=total_frequent_action_words,
    #     Bullets_Total=Bullets_Total,
    #     standard_bullet_flag=standard_bullet_flag,
    #     total_repeated_actionwords=total_frequent_action_words,
    #     ats_date=ats_date,
    #     dates_nonAts=dates_nonAts,
    #     clean_measurable=clean_measurable,
    #     total_word_count=total_word_count,
    #     email_finderSet=email_finderSet,
    #     phonenumbers_finderSet=phones,
    #     images=images,
    #     linkedIn_flag=linkedIn_flag,
    #     pages_count=pages_count,
    #     personalPronouns=personalPronouns,
    #     tables_flag=tables_flag,
    #     Work_Project_Headings=Work_Project_Headings,
    #     EduSkill_Headings=EduSkill_Headings,
    #     NRlength=NRlength,
    #     ORlength_db=ORlength_db,
    #     Skills_Total=Skills_Total,
    #     standard_match_headings_count=standard_match_headings_count,
    #     sectionMapCount=sectionMapCount,
    #     phone_all1=all_phones,
    #     actualHeadingsCount=actualHeadingsCount,
    #     experience=experience,
    #     fillerwords_total=total_filler_words,
    #     output_voice=voice,
    #     # repeated_words=repeated_words,
    #     repeated_words=repeated_words_new,
    #     file_name=file_name,
    #     name_output=name_output,
    #     ORlength=ORlength 
    # )

    # print("📊 Final Resume Score:", resume_score)
    # print("📊 Final Resume Score Array:", score_array)

    # save outputs to a file
    output_data = {
        "File Name": file_name,
        # "headings": headings,
        "subHeadings": subHeadings,
        # "notRequired_Heading": notRequired_Heading,
        # "standard_match_headings": standard_match_headings,
        "is_multi_column": is_multi_column,
        "has_same_font": has_same_font_size,
        "same_font_size": same_font_size,
        "cleaned_headers": cleaned_headers,
        # "standard_headings_map_groq": standard_headings_map_groq,
        # "standard_headings_map": standard_headings_map,
        # "sh_map_scores": valhalla_standard_headers,
        # "sections": sections,
        # "standard_sections": standard_sections,
        "max_recurring_font_size": max_recurring_font_size,
        "max_font_headers": str(max_font_size_headers),
        "max_font_headers_count": len(max_font_size_headers),
        "different_font_headers": str(different_font_size_headers),
        "different_font_headers_count": len(different_font_size_headers),

        "max_font_missing_headers": str(max_font_missing_headers),
        "max_font_missing_headers_count": len(max_font_missing_headers),
        "different_font_cleaned_headers": str(different_font_cleaned_headers),
        "different_font_cleaned_headers_count": len(different_font_cleaned_headers),
        "matched_headers_from_groq_input": str(matched_headers_from_groq_input),
        "matched_headers_from_groq_input_count": len(matched_headers_from_groq_input),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        # "action_words": action_words,
        # "total_action_words": total_action_words,
        # "all_action_words": all_action_words,
        # "frequent_action_words": frequency_list,
        # "total_frequent_action_words": total_frequent_action_words,
        # "repeated_frequency": repeated_frequency,
        # "negative_action_words": negative_action_words,
        # "total_negative_action_words": total_negative_action_words,
        # "all_negative_action_words": all_negative_action_words,
        # "frequencyList_negative": frequencyList_negative,
        # "total_repeated_actionwords_negative": total_repeated_actionwords_negative,
        # "repeated_frequency_negative": repeated_frequency_negative,
        # "filler_words": filler_words,
        # "total_filler_words": total_filler_words,
        # "all_filler_words": all_filler_words,
        # "passive_voice_constructions": voice,
        # "font_styles": font_styles,
        # "standard_font_style_flag": standard_font_style_flag,
        # "multiple_font_style": multiple_font_style,
        # "font_sizes": font_sizes,
        # "multiple_font_size": multiple_font_size,
        # "phone_numbers": all_phones,
        # "reg_Phone": reg_Phone,
        # "email_finderSet": list(email_finderSet),
        # "url": url,
        # "linkedIn_flag": linkedIn_flag,
        # "url_flag": url_flag,
        # "images_found": images,
        # "finalBullet": finalBullet,
        # "Bullets_Total": Bullets_Total,
        # "standard_bullet_flag": standard_bullet_flag,
        # "ats_date": ats_date,
        # "dates_nonAts": dates_nonAts,
        # "measurable_ner": measurable,
        # "clean_measurable": clean_measurable,
        # "repeated_words": repeated_words_new,
        # "skills": skills,
        # "Skills_Total": Skills_Total,
        # # "tables": tables,
        # # "tables2": tables2,
        # # "tables_flag": tables_flag,
        # # "file_size_kb": file_size_kb,
        # # "file_type": file_type,
        # # "flag_file_size": flag_file_size,
        # "max_size": max_size,
        # "content_size_flag": content_size_flag,
        # "font_colors": font_colors,
        # "font_colors_Total": font_colors_Total,
        # "standard_color_flag": standard_color_flag,
        # "total_word_count": total_word_count,
        # "pages_count": pages_count,
        # "personalPronouns": personalPronouns,
        # "score": resume_score,
    }

    # with open(FINAL_OUTPUT_FILE_PATH, "w") as f:
    #     json.dump(output_data, f, indent=4)

    return output_data

# def process_multiple_resume():

#     analysis_results = []

#     for file_name in os.listdir(DATA_DIR):

#         if file_name.endswith(".pdf"):

#             start_time = time.time()

#             print(f"🔍 Processing resume: {file_name}")
            
#             pdf_path = os.path.join(DATA_DIR, file_name)
#             pdf_bytes = load_local_pdf(pdf_path)            
#             result_doc = process_resume(pdf_bytes, pdf_path, experience=3, file_name=file_name)
#             analysis_results.append(result_doc)

#             print(f"📊 Resume processed: {file_name}")
#             print(f"📊 Time taken: {time.time() - start_time:.2f} seconds")


#     return analysis_results

# # with batching
# def process_multiple_resume():
#     analysis_results = []
#     batch_count = 1  # to name output files

#     for idx, file_name in enumerate(os.listdir(DATA_DIR), start=1):

#         if file_name.endswith(".pdf"):

#             start_time = time.time()

#             print(f"🔍 Processing resume: {file_name}")
            
#             pdf_path = os.path.join(DATA_DIR, file_name)
#             pdf_bytes = load_local_pdf(pdf_path)            
#             result_doc = process_resume(
#                 pdf_bytes, 
#                 pdf_path, 
#                 experience=3, 
#                 file_name=file_name
#             )

#             analysis_results.append(result_doc)

#             print(f"📊 Resume processed: {file_name}")
#             print(f"📊 Time taken: {time.time() - start_time:.2f} seconds")

#             # ---- Save batch every 10 resumes ----
#             if idx % BATCH_SIZE == 0:
#                 output_file = os.path.join(
#                     OUTPUT_DIR, 
#                     f"results_batch_{batch_count}.csv"
#                 )
#                 pd.DataFrame(analysis_results).to_csv(output_file, index=False)

#                 print(f"💾 Saved batch {batch_count} to {output_file}")

#                 analysis_results = []  # reset batch
#                 batch_count += 1

#     # ---- Save remaining results ----
#     if analysis_results:
#         output_file = os.path.join(
#             OUTPUT_DIR, 
#             f"results_batch_{batch_count}.csv"
#         )
#         pd.DataFrame(analysis_results).to_csv(output_file, index=False)

#         print(f"💾 Saved final batch {batch_count} to {output_file}")

#     return

# START_FROM = 140  # already processed
START_FROM = 150  # already processed

def process_multiple_resume():
    analysis_results = []
    batch_count = (START_FROM // BATCH_SIZE) + 1  # continue batch numbering

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    pdf_files.sort()  # IMPORTANT: keep order consistent

    pdf_files = pdf_files[:200]

    for idx, file_name in enumerate(pdf_files, start=1):

        if idx <= START_FROM:
            continue  # skip already processed files

        start_time = time.time()

        print(f"🔍 Processing resume: {file_name}")
        
        pdf_path = os.path.join(DATA_DIR, file_name)
        pdf_bytes = load_local_pdf(pdf_path)            
        result_doc = process_resume(
            pdf_bytes, 
            pdf_path, 
            experience=3, 
            file_name=file_name
        )

        analysis_results.append(result_doc)

        print(f"📊 Resume processed: {file_name}")
        print(f"📊 Time taken: {time.time() - start_time:.2f} seconds")

        # batching based on NEW processed count
        if len(analysis_results) == BATCH_SIZE:
            output_file = os.path.join(
                OUTPUT_DIR, 
                f"results_batch_{batch_count}.csv"
            )
            pd.DataFrame(analysis_results).to_csv(output_file, index=False)

            print(f"💾 Saved batch {batch_count} to {output_file}")

            analysis_results = []
            batch_count += 1

    # save remaining
    if analysis_results:
        output_file = os.path.join(
            OUTPUT_DIR, 
            f"results_batch_{batch_count}.csv"
        )
        pd.DataFrame(analysis_results).to_csv(output_file, index=False)

        print(f"💾 Saved final batch {batch_count} to {output_file}")

if __name__ == "__main__":

    # For local testing    
    # SINGLE RESUME
    file_name = "Eklavya_Resume26.pdf"
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

    output = process_resume(pdf_bytes, pdf_path, experience, file_name)


    # BULK RESUME

    # without batching
    # start_time = time.time()
    # analysis_results = process_multiple_resume()

    # pd.DataFrame(analysis_results).to_csv("new_prompt_model_metrics_2704.csv", index=False)
    # print(f"📊 Total time taken: {time.time() - start_time:.2f} seconds")

    # with batching

    # start_time = time.time()
    # process_multiple_resume()
    # print(f"📊 Total time taken: {time.time() - start_time:.2f} seconds")