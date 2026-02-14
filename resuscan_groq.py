from font_based_line_grouping import filter_body_content
from doclayout_yolo import YOLOv10
from urllib.parse import urlparse
from io import BytesIO
from groq import Groq
import boto3
import json
import time
import os
import fitz
import uuid

IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
DEFAULT_OUTPUT_DIR = "/tmp" if IS_LAMBDA else os.getcwd()

# Load .env only if running locally
if os.getenv("AWS_LAMBDA_FUNCTION_NAME") is None:
    from dotenv import load_dotenv
    load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Load YOLO model globally
MODEL_PATH = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
MODEL = YOLOv10(MODEL_PATH)
CLASS_NAMES = MODEL.names


def extract_with_doclayout_yolo(pdf_bytes, dpi=300, conf=0.15):
    """Extract text from PDF using DocLayout YOLO"""
    
    # Create temp directory
    temp_dir = os.path.join(DEFAULT_OUTPUT_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Convert PDF pages to images
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_info = []
        
        for i, page in enumerate(doc):
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_path = os.path.join(temp_dir, f"page_{i+1}.png")
            pix.save(img_path)
            pages_info.append((img_path, pix.width, pix.height, page.rect.width, page.rect.height))
        
        # Run YOLO detection and extract text
        detected_blocks = []
        
        for page_idx, (img_path, img_w, img_h, pdf_w, pdf_h) in enumerate(pages_info):
            results = MODEL.predict(source=img_path, imgsz=1024, conf=conf, device="cpu")
            page = doc[page_idx]
            
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls)
                    class_name = CLASS_NAMES[class_id]
                    coords = box.xyxy.cpu().numpy()[0]
                    
                    # Convert pixel coordinates to PDF coordinates
                    x1, y1, x2, y2 = coords
                    rect = fitz.Rect(
                        x1 * pdf_w / img_w,
                        y1 * pdf_h / img_h,
                        x2 * pdf_w / img_w,
                        y2 * pdf_h / img_h,
                    )
                    
                    text = page.get_textbox(rect).strip()
                    
                    if text:
                        detected_blocks.append({
                            "page": page_idx,
                            "class_name": class_name,
                            "text": text,
                            "y": rect.y0,
                            "x": rect.x0
                        })
        
        doc.close()
        
        # Sort blocks by page, x, and y
        # detected_blocks.sort(key=lambda b: (b["page"], b["x"] > (pdf_w / 3), b["y"]))
        detected_blocks.sort(key=lambda b: (b["page"], b["x"] > (pdf_w / 3), b["y"]))
        
        # to return blocks with entire info
        # return detected_blocks
        
        # Extract only the text from sorted blocks
        extracted_texts = [block["text"] for block in detected_blocks]
        
        return extracted_texts

    finally:
        # Cleanup temp files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def extract_original_text_linewise(pdf_bytes):
    """Extract original text line by line using PyMuPDF (fitz)"""
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_lines = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        lines = text.split('\n')
        
        for line in lines:
            if line.strip():  # Only add non-empty lines
                all_lines.append({
                    "page": page_num,
                    "text": line.strip()
                })
    
    doc.close()

    extracted_texts = [line["text"] for line in all_lines]

    return extracted_texts


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
    
    # Extract token usage
    token_usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

    return cleaned_headers, token_usage


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

        cleaned_headers, token_usage = filter_headers_with_groq(linewise_content_with_fonts, prompt)

        # Extract with YOLO
        yolo_blocks = extract_with_doclayout_yolo(pdf_bytes)
        
        # Extract original text
        original_lines = extract_original_text_linewise(pdf_bytes)

        results = {
            "Line wise content with fonts": linewise_content_with_fonts,
            "seperator1": "\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n",
            "Fonts and number of words": font_and_words,
            "seperator2": "\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n",
            "Body font size": max_font_size,
            "seperator3": "\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n",
            "Body words": max_words,
            "seperator4": "\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n",
            "Cleaned headers from Groq": cleaned_headers,
            "seperator5": "\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n",
            "Groq Token Usage": token_usage,
            "seperator6": "\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n",
            # "YOLO Detected Blocks": yolo_blocks,
            "YOLO Detected Blocks": "\n".join(yolo_blocks),
            "seperator7": "\n\n-------------------------------------------------------------------------------------------------------------------------------------------\n\n",
            # "Original Text Line by Line": original_lines
            "Original Text Line by Line": "\n".join(original_lines)
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

    pdf_file_path = "TANVI GAWALI CV.pdf"

    pdf_file = load_local_pdf(pdf_file_path)

    # 1. Extract with DocLayout YOLO
    print("\n" + "="*80)
    print("📄 DOCLAYOUT YOLO EXTRACTION")
    print("="*80)
    yolo_blocks = extract_with_doclayout_yolo(pdf_file)
    for block in yolo_blocks:
        # print(f"[Page {block['page']}] [{block['class_name']}]: {block['text']}")
        print(block['text'])
    
    # 2. Extract original text line by line
    print("\n" + "="*80)
    print("📄 ORIGINAL TEXT LINE BY LINE")
    print("="*80)
    original_lines = extract_original_text_linewise(pdf_file)
    for line in original_lines:
        # print(f"[Page {line['page']}]: {line['text']}")
        print(line['text'])
    
    # 3. Existing font-based extraction
    print("\n" + "="*80)
    print("📄 FONT-BASED EXTRACTION")
    print("="*80)
    linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_file)

    prompt_template = """
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
    cleaned_headers, token_usage = filter_headers_with_groq(
        linewise_content_with_fonts,
        prompt_template
    )

    print("\n✅ Cleaned Resume Headers from LLaMA:")
    print(cleaned_headers)
    
    print("\n📊 Groq Token Usage:")
    print(f"  Input tokens: {token_usage['input_tokens']}")
    print(f"  Output tokens: {token_usage['output_tokens']}")
    print(f"  Total tokens: {token_usage['total_tokens']}")