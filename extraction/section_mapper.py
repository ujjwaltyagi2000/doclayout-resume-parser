import ast
import re

# Original function without filtering by class_id
# def combine_section_text(section_data):
#     texts = []

#     for item in section_data:
#         raw = item.get("text", "")

#         # clean
#         cleaned = raw.replace("\u2022", "")  # remove bullets
#         cleaned = re.sub(r"\s+", " ", cleaned).strip()

#         if cleaned:
#             texts.append(cleaned)

#     return "\n".join(texts)

# # Updated function to only include List-item (class_id = 3) and clean text accordingly
# def combine_section_text(section_data):
#     texts = []

#     for item in section_data:
#         # ✅ Only take List-item (class_id = 3)
#         if item.get("class_id") != 3:
#             continue

#         raw = item.get("text", "")

#         # ✅ Clean bullets and formatting
#         cleaned = raw.replace("\u2022", "")  # remove bullet symbol
#         cleaned = cleaned.replace("\n", " ")  # remove line breaks inside text
#         cleaned = re.sub(r"\s+", " ", cleaned).strip()  # normalize spaces

#         if cleaned:
#             texts.append(cleaned)

#     # ✅ Each bullet on new line
#     return "\n".join(texts)

import re
import ast

BULLET_PATTERNS = [
    r'^\s*[\u2022\u2023\u25E6\u2043\u2219]',
    r'^\s*[-*•●◦▪►➤➢➣➥➦➧➨➩➪➫➭➮➯]',
    r'^\s*\d+\.',
    r'^\s*[a-zA-Z]\)',
    r'^\s*\(\d+\)',
]

MULTI_BULLET_SPLIT = r'[\u2022•\-\*➤➢➣➤]+'


def clean_bullet_text(text):
    if not text:
        return ""

    text = text.replace("\n", " ")

    for pattern in BULLET_PATTERNS:
        text = re.sub(pattern, '', text)

    text = re.sub(r'^[^\w]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def combine_section_text(section_data):
    texts = []

    for item in section_data:
        if item.get("class_id") != 3:
            continue

        raw = item.get("text", "")

        # split_parts = re.split(MULTI_BULLET_SPLIT, raw)

        # for part in split_parts:
        #     cleaned = clean_bullet_text(part)

        #     if cleaned:
        #         texts.append(cleaned)
        cleaned = clean_bullet_text(raw)

        if cleaned:
            texts.append(cleaned)
            
    # return combined string + list of bullet points
    combined_string = "\n".join(texts)
    return combined_string, texts


def map_content_to_standard_header(sections, standard_headers):

    if isinstance(standard_headers, str):
        standard_headers = ast.literal_eval(standard_headers)

    mapped_output = {}

    for std_header, actual_header in standard_headers.items():

        print(f"\n🔹 Processing: {std_header} → {actual_header}")

        if actual_header and actual_header in sections:
            
            combined_text, bullet_list = combine_section_text(sections[actual_header])

            mapped_output[std_header] = {
                "text": combined_text,
                "bullets": bullet_list
            }

            print(f"✅ Mapped + Combined content from '{actual_header}'")

        else:
            mapped_output[std_header] = {
                "text": "",
                "bullets": []
            }

            print(f"⚠️ No content found")

    return mapped_output
# def map_content_to_standard_header(sections, standard_headers):

#     if isinstance(standard_headers, str):
#         standard_headers = ast.literal_eval(standard_headers)
#         print("Standard Headings converted from string to dict.")

#     mapped_output = {}

#     for std_header, actual_header in standard_headers.items():

#         print(f"\n🔹 Processing: {std_header} → {actual_header}")

#         if actual_header and actual_header in sections:
            
#             combined_text = combine_section_text(sections[actual_header])

#             mapped_output[std_header] = combined_text
#             print(f"✅ Mapped + Combined content from '{actual_header}'")

#         else:
#             mapped_output[std_header] = ""
#             print(f"⚠️ No content found")

#     return mapped_output