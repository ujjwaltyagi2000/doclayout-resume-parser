"""

Script to extract headers from a resume PDF file using docling and further filter relevant ones using Groq API.

Status: Working ✅

"""

from docling.document_converter import DocumentConverter
from urllib.parse import urlparse
from io import BytesIO
from groq import Groq
import boto3
import time
import json
import os

IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
DEFAULT_OUTPUT_DIR = "/tmp" if IS_LAMBDA else os.getcwd()

import os

# Load .env only if running locally
if os.getenv("AWS_LAMBDA_FUNCTION_NAME") is None:
    from dotenv import load_dotenv
    load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
converter = DocumentConverter()


# STEP 1: Extract Headers

def get_headers_from_pdf_bytes(pdf_bytes: bytes):

    temp_path = f"{DEFAULT_OUTPUT_DIR}/resume.pdf"

    # Write PDF to Lambda temp storage
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    result = converter.convert(temp_path)
    doc = result.document

    headers = []

    for item, level in doc.iterate_items():
        label = item.label.value if hasattr(item.label, 'value') else str(item.label)
        text = item.text.strip() if hasattr(item, 'text') else ""

        size = 0
        if hasattr(item, 'prov') and item.prov:
            bbox = item.prov[0].bbox
            size = round(bbox.t - bbox.b, 2)

        if label == "section_header" and text:
            headers.append({
                "text": text,
                "level": level,
                "font_size": size
            })

    print("\n🔎 Extracted Headers:")
    print(headers)

    return headers



# STEP 2: Send to Groq LLaMA

def filter_headers_with_groq(headers):

    prompt = f"""
    Extract valid resume section headers from this list: {headers}

    Each List Item contains three fields: 
    1. Header Text
    2. Header Level (within document Hierarchy)
    3. Font Size

    Rules:
    - Keep only standard resume sections (e.g., Education, Experience, Skills, Projects, etc.)
    - Remove names, dates, company names, and project details
    - Remove duplicates
    - Return ONLY a Python list, nothing else

    IMPORTANT: Your entire response must be ONLY the list in this exact format:
    ['Header1', 'Header2', 'Header3']

    Do not include explanations, code, markdown, or any other text.
    """


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    cleaned_headers = response.choices[0].message.content

    print("\n✅ Cleaned Resume Headers from LLaMA:")
    print(cleaned_headers)

    return cleaned_headers

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

# LOCAL TESTING
def get_headers_from_pdf_path(pdf_path: str):

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    headers = []

    for item, level in doc.iterate_items():
        label = item.label.value if hasattr(item.label, 'value') else str(item.label)
        text = item.text.strip() if hasattr(item, 'text') else ""

        size = 0
        if hasattr(item, 'prov') and item.prov:
            bbox = item.prov[0].bbox
            size = round(bbox.t - bbox.b, 2)

        if label == "section_header" and text:
            headers.append({
                "text": text,
                "level": level,
                "font_size": size
            })

    print("\n🔎 Extracted Headers:")
    print(headers)

    return headers

def run_local(pdf_path: str):

    print("📄 Running locally with:", pdf_path)

    headers = get_headers_from_pdf_path(pdf_path)
    cleaned_headers = filter_headers_with_groq(headers)

    return {
        "docling_headers": headers,
        "groq_headers": cleaned_headers
    }

def handler(event, context):
    
    start_time = time.time()

    try:
        print("✅ Lambda invoked")

        req_body = json.loads(event["body"]) if "body" in event else event

        aws_access_key = req_body["aws_access_key"]
        aws_secret_key = req_body["aws_secret_key"]
        pdf_url = req_body["pdf_url"]


        pdf_bytes = fetch_pdf_from_s3(
            pdf_url,
            aws_access_key,
            aws_secret_key
        )

        headers = get_headers_from_pdf_bytes(pdf_bytes)

        cleaned_headers = filter_headers_with_groq(headers)

        results = {
            "docling_headers": headers,
            "seperator": """

            
-------------------------------------------------------------------------------------------------------------------------------------------


""",
            "groq_headers": cleaned_headers
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

    pdf_path = "resume.pdf"
    response = run_local(pdf_path)

    print("📃 Final Response: \n", response)