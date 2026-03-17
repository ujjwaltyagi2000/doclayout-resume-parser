from font_based_line_grouping import filter_body_content, get_full_content_with_fonts
from transformer_inference import run_transformer_mapping
from fuzzy_inference import run_fuzzy_mapping
from urllib.parse import urlparse
from io import BytesIO
from groq import Groq
import boto3
import json
import time
import ast
import os

IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
DEFAULT_OUTPUT_DIR = "/tmp" if IS_LAMBDA else os.getcwd()


# Load .env only if running locally
if os.getenv("AWS_LAMBDA_FUNCTION_NAME") is None:
    from dotenv import load_dotenv
    load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def filter_headers_with_groq(headers, prompt_template):

    # Inject headers into the prompt template
    prompt = prompt_template.format(headers=headers)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    cleaned_headers = response.choices[0].message.content

    return cleaned_headers

def get_standard_headings_map(cleaned_headers, prompt_template):

    # Inject headers into the prompt template
    prompt = prompt_template.format(cleaned_headers=cleaned_headers)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    standard_headings_map = response.choices[0].message.content

    return standard_headings_map

def fetch_pdf_from_s3(pdf_url: str, aws_access_key: str, aws_secret_key: str) -> bytes:
    parsed_url = urlparse(pdf_url)
    bucket_name = parsed_url.netloc.split(".")[0]
    region = parsed_url.netloc.split(".")[2]
    key = parsed_url.path.lstrip("/")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region
    )

    response = s3.get_object(Bucket=bucket_name, Key=key)
    return response["Body"].read()

def load_local_pdf(pdf_path: str) -> bytes:
    with open(pdf_path, "rb") as f:
        return f.read()

def normalize(text):
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def find_headers(linewise_content, cleaned_headers):
    matched_headers = []

    for line in linewise_content:
        line_text = normalize(line['text'])

        for header in cleaned_headers:
            header_text = normalize(header)

            if header_text in line_text:
                matched_headers.append({
                    "header": header,
                    "y_position": line["y_position"],
                    "font": line["font"]
                })
                break

    # sort top → bottom
    # matched_headers = sorted(matched_headers, key=lambda x: x["y_position"])
    matched_headers = sorted(matched_headers, key=lambda x: x["y_position"], reverse=True)

    return matched_headers

def build_sections_from_positions(full_content, matched_headers):
    sections = {}

    if not matched_headers:
        return {"Full Document": full_content}

    for i, header_info in enumerate(matched_headers):
        header = header_info["header"]
        start_y = header_info["y_position"]

        # next header ka y (end boundary)
        end_y = (
            matched_headers[i + 1]["y_position"]
            if i + 1 < len(matched_headers)
            else float("inf")
        )

        section_lines = []

        # for line in full_content:
        #     y = line["y_position"]

        #     if start_y < y < end_y:
        #     # if end_y < y < start_y:
        #         section_lines.append(line)

        for line in full_content:
            y = line["y_position"]
            if end_y < y < start_y:  # reverse the comparison
                section_lines.append(line)

        sections[header] = section_lines

    return sections

def handler(event, context):
    
    start_time = time.time()

    try:
        print("✅ Lambda invoked")

        req_body = json.loads(event["body"]) if "body" in event else event

        aws_access_key = req_body["aws_access_key"]
        aws_secret_key = req_body["aws_secret_key"]
        pdf_url = req_body["pdf_url"]
        cleaned_headers_prompt = req_body["prompt"]
        standard_headings_prompt = req_body["shPrompt"]

        pdf_bytes = fetch_pdf_from_s3(
            pdf_url,
            aws_access_key,
            aws_secret_key
        )

        linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_bytes)

        # print(f"✅ Body Font Size: {max_size}, Words: {len(max_words)}")

        cleaned_headers = filter_headers_with_groq(linewise_content_with_fonts, cleaned_headers_prompt)

        print(f"📃 Cleaned Headers: {cleaned_headers}")

        standard_headings_map = get_standard_headings_map(cleaned_headers, standard_headings_prompt)
        print(f"📃 Standard Headers: {standard_headings_map}")
        # headers = get_headers_from_pdf_bytes(pdf_bytes)

        # Convert cleaned_headers string to Python list
        cleaned_headers_list = ast.literal_eval(cleaned_headers)
        # cleaned_headers_list = ast.literal_eval(cleaned_headers)

        # Build Sections
        # sections = build_sections(full_resume_content, cleaned_headers_list)

        matched_headers = find_headers(linewise_content_with_fonts, cleaned_headers_list)

        print(f"📃 Matched Headers: {matched_headers}")

        sections = build_sections_from_positions(full_resume_content, matched_headers)

        import json
        print(json.dumps(sections, indent=2))

        with open("sections.json", "w") as f:
            json.dump(sections, f, indent=2)

        # Run transformer based mapper
        transformer_output = run_transformer_mapping(cleaned_headers_list)

        print(f"📃 Standard Headers from Transformer: {transformer_output}")

        # Run fuzzy mapping
        fuzzy_output = run_fuzzy_mapping(cleaned_headers_list)

        print(f"📃 Standard Headers from Fuzzy: {fuzzy_output}")


        results = {
            "Line wise content with fonts": linewise_content_with_fonts,
            "seperator1": """

            
-------------------------------------------------------------------------------------------------------------------------------------------


""",
            "Fonts and number of words": font_and_words,
            "seperator2": """

            
-------------------------------------------------------------------------------------------------------------------------------------------


""",
            "Body font size": max_font_size,
            "seperator3": """

            
-------------------------------------------------------------------------------------------------------------------------------------------


""",
            "Body words": max_words,
            "seperator4": """

            
-------------------------------------------------------------------------------------------------------------------------------------------


""",
            "Cleaned headers from Groq": cleaned_headers,
            "seperator5": """

            
-------------------------------------------------------------------------------------------------------------------------------------------


""",
            "Standard headings map from Groq": standard_headings_map,
            "seperator6": """

            
-------------------------------------------------------------------------------------------------------------------------------------------


""",
            "Standard headings map from Transformer": transformer_output,
            "seperator7": """

            
-------------------------------------------------------------------------------------------------------------------------------------------


""",        
            "Standard headings map from Fuzzy": fuzzy_output["canonical"]
        }

        print("✅ Extraction completed")
        print(f"📃 Final Response: \n {results}")

        # with open("output.json", "w") as f:
            # json.dump(results, f, indent=4)

        return {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization"
                },
                "body": json.dumps(results)
            }


    except Exception as e:
        print("❌ Error:", str(e))
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            },
            "body": json.dumps({"error": str(e)}),
        }

    finally:
        print(f"⌚ Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":

    # pdf_file_path = "Puunita Chaturvedi.pdf"
    # pdf_file_path = "resume.pdf"
    # pdf_file_path = "Ujjwal Tyagi.pdf"
    # pdf_file_path = "TANVI GAWALI CV.pdf"
    # pdf_file_path = "resume/Megha resume.pdf"
    # pdf_file_path = "resume/Puunita Chaturvedi.pdf"
    pdf_file_path = "Soloman Kadam.pdf"

    print(f"📁 File Name: {pdf_file_path}")
    pdf_file = load_local_pdf(pdf_file_path)
    
    linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_file)

    full_resume_content = get_full_content_with_fonts(pdf_file)

    print(full_resume_content)
    
    # Print line by line
    # print(lines)

    # Remove y_position (can try to include as well in an example)
    # filtered_resume_text = [
    #     {k: v for k, v in item.items() if k != 'y_position'}
    #     for item in lines
    # ]

    # 🚀 Prompt for getting clean headers
    cleaned_headers_prompt_template = """
        You are a resume expert. 
        Extract valid resume section headers from this list: {headers}

        Each List Item contains three fields: 
        1. Header Text
        3. Font Size
        2. Y co-ordinate (within document)

        Rules:
        - Keep only standard resume sections (e.g., Education, Experience, Skills, Projects, etc.)
        - Remove names, dates, company names, and project details
        - Remove duplicates
        - Return ONLY a Python list, nothing else
        - If there are no valid headers, return an empty list

        IMPORTANT: Your entire response must be ONLY the list in this exact format:
        ['Header1', 'Header2', 'Header3']

        Do not include explanations, code, markdown, or any other text.
    """

    print(f"📃 Unfiltered Headers: {linewise_content_with_fonts}")
    # Feed extracted content to Groq
    cleaned_headers = filter_headers_with_groq(
        linewise_content_with_fonts,
        cleaned_headers_prompt_template
    )

    print("\n✅ Cleaned Resume Headers from LLaMA:")
    print(cleaned_headers)

    cleaned_headers_list = ast.literal_eval(cleaned_headers)

    # Build Sections
    # sections = build_sections(full_resume_content, cleaned_headers_list)
    matched_headers = find_headers(linewise_content_with_fonts, cleaned_headers_list)

    print("\n✅ Matched Resume Headers:")
    print(matched_headers)

    sections = build_sections_from_positions(full_resume_content, matched_headers)

    # import json
    # print(json.dumps(sections, indent=2))

    with open("sections.json", "w") as f:
        json.dump(sections, f, indent=2)

    # ✅ More precise prompt
    standard_headings_prompt = f"""

    You are a resume expert. I will provide you with a set of headers that are extracted from a resume.

    Here are the extracted headers: {cleaned_headers}
    
    There are multiple ways to write the same thing. Example: Professional Experience, Work Experience, Work history, all can be called "Experience".

    Here is a list of Standard Headers: ["Objective", "Summary", "Experience", "Projects"]  
    
    I want you map the extract headers to the Standard Headers as key value pairs where each standard header is a key and the extracted header is the value.
    
    Example 1: if you get a heading Professional Summary, I want you to map it to "Summary". 
    Example 2: if you get a heading Work Experience, I want you to map it to "Experience".


    Your task is to return a python dictionary where the keys are the standard headers and the values are the extracted headers. 

    CRITICAL RULES:

    1. Do not create other keys, only find matches for the provided keys.\
    
    2. If a key doesn't have a match, set it equal to an empty string ""

    3. If no match is found for any standard header, return an empty dictionary.

    Do not include explanations, code, markdown, or any other text.

"""
    standard_headings_map = get_standard_headings_map(
        cleaned_headers, 
        standard_headings_prompt
    )

    print("\n✅ Standard headings map from Llama:")
    print(standard_headings_map)
    
    # # Convert cleaned_headers string to Python list
    # cleaned_headers_list = ast.literal_eval(cleaned_headers)

    # # Run transformer based mapper
    # transformer_output = run_transformer_mapping(cleaned_headers_list)

    # print(f"📃 Standard Headers from Transformer: {transformer_output}")

    # # Run fuzzy based mapper
    # fuzzy_output = run_fuzzy_mapping(cleaned_headers_list)

    # print(f"📃 Standard Headers from Fuzzy: {fuzzy_output}")