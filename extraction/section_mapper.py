import ast

import re

def combine_section_text(section_data):
    texts = []

    for item in section_data:
        raw = item.get("text", "")

        # clean
        cleaned = raw.replace("\u2022", "")  # remove bullets
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if cleaned:
            texts.append(cleaned)

    return "\n".join(texts)

import ast

def map_content_to_standard_header(sections, standard_headers):

    if isinstance(standard_headers, str):
        standard_headers = ast.literal_eval(standard_headers)
        print("Standard Headings converted from string to dict.")

    mapped_output = {}

    for std_header, actual_header in standard_headers.items():

        print(f"\n🔹 Processing: {std_header} → {actual_header}")

        if actual_header and actual_header in sections:
            
            combined_text = combine_section_text(sections[actual_header])

            mapped_output[std_header] = combined_text
            print(f"✅ Mapped + Combined content from '{actual_header}'")

        else:
            mapped_output[std_header] = ""
            print(f"⚠️ No content found")

    return mapped_output