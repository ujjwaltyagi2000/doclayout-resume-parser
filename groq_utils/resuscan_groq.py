from info.font_based_line_grouping import filter_body_content
from urllib.parse import urlparse
from io import BytesIO
from groq import Groq
import boto3
import json
import time
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

def handler(event, context):
    
    start_time = time.time()

    try:
        print("✅ Lambda invoked")

        req_body = json.loads(event["body"]) if "body" in event else event

        aws_access_key = req_body["aws_access_key"]
        aws_secret_key = req_body["aws_secret_key"]
        pdf_url = req_body["pdf_url"]
        prompt = req_body["prompt"]

        pdf_bytes = fetch_pdf_from_s3(
            pdf_url,
            aws_access_key,
            aws_secret_key
        )

        linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_bytes)

        # print(f"✅ Body Font Size: {max_size}, Words: {len(max_words)}")

        cleaned_headers = filter_headers_with_groq(linewise_content_with_fonts, prompt)


        # headers = get_headers_from_pdf_bytes(pdf_bytes)


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
            "Cleaned headers from Groq": cleaned_headers
        }

        print("✅ Extraction completed")
        print(f"📃 Final Response: \n {results}")

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
    pdf_file_path = "TANVI GAWALI CV.pdf"
    # pdf_file_path = "Megha resume.pdf"

    pdf_file = load_local_pdf(pdf_file_path)

    linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_file)

    # Print line by line
    # for line in linewise_content_with_fonts:
    #     print(line)

    # Remove y_position (can try to include as well in an example)
    # filtered_resume_text = [
    #     {k: v for k, v in item.items() if k != 'y_position'}
    #     for item in lines
    # ]

    # Placeholder prompt template for local testing
    prompt_template = prompt = """
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

    IMPORTANT: Your entire response must be ONLY the list in this exact format:
    ['Header1', 'Header2', 'Header3']

    Do not include explanations, code, markdown, or any other text.
"""
    # Feed extracted content to Groq
    cleaned_headers = filter_headers_with_groq(
        linewise_content_with_fonts,
        prompt_template
    )

    print("\n✅ Cleaned Resume Headers from LLaMA:")
    print(cleaned_headers)
