"""
Built over ordered_classes_within_sections.py.

This module: 
    1. imports filter_body_content and filter_headers_with_groq function from resuscan_groq
    2. the above functions return cleaned headers from the resume.
    3. runs yolo over the document and maps extracted headings from cleaned headers to their corresponding bounding boxes
    4. find font sizes of yolo extracted headings
    5. find headers with the same font size as resuscan headings
    6. builds sections from extracted headings by sorting the bounding boxes along x-axis, followed by y-axis
    7. this enables multi-column resume parsing
    8. within section headers, text is now in the same ordered (coupled with their class names) as in the original resume

Status: Working ✅
"""

from doclayout_yolo import YOLOv10
# from resuscan_fonts import get_headings, get_heading_font_sizes  # Added get_heading_font_sizes
from resuscan_groq import filter_body_content, filter_headers_with_groq
from urllib.parse import urlparse
from collections import defaultdict

import boto3
import fitz
import json
import time
import os
import uuid

# =========================
# Global model (loaded once per container)
# =========================
MODEL_PATH = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
DEFAULT_OUTPUT_DIR = "/tmp" if IS_LAMBDA else os.getcwd()
MODEL = YOLOv10(MODEL_PATH)
LOCAL_OUTPUT_DIR = "json"
LOCAL_OUTPUT_JSON_FILE_NAME = "test.json"
LOCAL_OUTPUT_FILE_PATH = os.path.join(LOCAL_OUTPUT_DIR, LOCAL_OUTPUT_JSON_FILE_NAME)
CLASS_NAMES = MODEL.names

# =========================
# Layout extractor
# =========================
class LayoutClassExtractor:
    def __init__(self, pdf_bytes, dpi=300, conf=0.15):
        # -------------------------
        # Step 1: Get headings via Groq (LLM filtering)
        # -------------------------

        self.temp_dir = os.path.join("/tmp", str(uuid.uuid4()))
        # self.temp_dir = os.path.join(os.getcwd(), "saved_cv_pages")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.pdf_bytes = pdf_bytes
        self.dpi = dpi
        self.conf = conf

        self.model = MODEL
        self.class_names = CLASS_NAMES

        print("🔍 Extracting candidate headers using filter_body_content()")

        linewise_content_with_fonts, font_and_words, max_font_size, max_words = filter_body_content(pdf_bytes)
            # filter_body_content(self.pdf_bytes)


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

        # print(f"📃 LIKEWISE CONTENT WITH FONTS: \n{linewise_content_with_fonts}\n")

        cleaned_headers = filter_headers_with_groq(
            linewise_content_with_fonts,
            prompt_template
        )

        import ast

        if isinstance(cleaned_headers, str):
            cleaned_headers = ast.literal_eval(cleaned_headers)

        
        print(f"✅ Groq Cleaned Headers: {cleaned_headers}")

        self.sub_headings = {
            self._normalize(h) for h in cleaned_headers
        }

        print(f"🚀 Sub Headings: {self.sub_headings}")

        self.detected_blocks = []
        self.pages_info = self._pdf_to_images()

    # =========================
    # Utilities
    # =========================
    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.upper().split())

    def _pdf_to_images(self):
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        pages = []

        for i, page in enumerate(doc):
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            img_path = os.path.join(self.temp_dir, f"page_{i+1}.png")
            pix.save(img_path)

            pages.append(
                (img_path, pix.width, pix.height, page.rect.width, page.rect.height)
            )

        doc.close()
        return pages

    @staticmethod
    def _pixel_to_pdf_rect(box, img_w, img_h, pdf_w, pdf_h):
        x1, y1, x2, y2 = box
        return fitz.Rect(
            x1 * pdf_w / img_w,
            y1 * pdf_h / img_h,
            x2 * pdf_w / img_w,
            y2 * pdf_h / img_h,
        )

    # =========================
    # Main extraction
    # =========================
    def extract(self):
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")

        for page_idx, (img_path, img_w, img_h, pdf_w, pdf_h) in enumerate(self.pages_info):
            results = self.model.predict(source=img_path, imgsz=1024, conf=self.conf, device="cpu")
            page = doc[page_idx]

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls)
                    class_name = self.class_names[class_id]
                    coords = box.xyxy.cpu().numpy()[0]
                    
                    rect = self._pixel_to_pdf_rect(coords, img_w, img_h, pdf_w, pdf_h)
                    # text = page.get_text("text", clip=rect).strip()
                    text = page.get_textbox(rect).strip()
                    
                    if not text: continue

                    self.detected_blocks.append({
                        "page": page_idx,
                        "x": rect.x0, # Store X for column detection
                        "y": rect.y0,
                        "x1": rect.x1,  # Store full rect for font extraction
                        "y1": rect.y1,
                        "class_id": class_id,
                        "class_name": class_name,
                        "text": text
                    })

        doc.close()
        self._cleanup()

        # --- SMART SORTING FOR MULTI-COLUMN ---
        # 1. Sort by page first
        # 2. Group by column (approximate x0)
        # 3. Sort by y0 within columns
        self.detected_blocks.sort(key=lambda b: (b["page"], b["x"] > (pdf_w / 3), b["y"])) 
        # Note: 'b["x"] > (pdf_w / 3)' is a simple heuristic to separate a narrow left sidebar 
        # from a main right column.

        sections = self._build_sections()
        return self.build_final_output(sections)


    # =========================
    # Section builder
    # =========================
    def _build_sections(self):
        sections = {}
        current_section = None

        for block in self.detected_blocks:
            normalized_text = self._normalize(block["text"])

            # Detect section header
            if (
                block["class_id"] != 3 and
                normalized_text in self.sub_headings
            ):
                current_section = block["text"]
                sections[current_section] = []
                continue

            if not current_section:
                continue

            sections[current_section].append({
                "class_id": block["class_id"],
                "class_name": block["class_name"],
                "text": block["text"]
            })

        # Save for debugging
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, "section_header.json")
        with open(output_path, "w") as f:
            json.dump(sections, f, indent=2)

        return sections

    
    def _get_full_resume_text(self):
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        lines = []

        for page in doc:
            text = page.get_text("text")
            page_lines = [l.strip() for l in text.split("\n") if l.strip()]
            lines.extend(page_lines)

        doc.close()
        return lines


    def _build_class_wise_content(self):
        classes = defaultdict(list)

        for block in self.detected_blocks:
            key = f"{block['class_id']} - {block['class_name']}"
            classes[key].append(block["text"])

        return dict(classes)


    def build_final_output(self, sections):
        final_output = {
            "meta": {
                "dpi": self.dpi,
                "confidence": self.conf,
                "total_pages": len(self.pages_info),
            },

            "seperator1": """

    --------------------------------------------------------------------------------------------------------------------

    """,

            "sub_headings": sorted(set(self.sub_headings)),

            "seperator2": """

    --------------------------------------------------------------------------------------------------------------------

    """,

            "sections_by_header": sections,

            "seperator3": """

    --------------------------------------------------------------------------------------------------------------------

    """,

            "full_resume_text": self._get_full_resume_text()
        }

        return final_output



    # =========================
    # Cleanup
    # =========================
    def _cleanup(self):
        try:
            if os.path.exists(self.temp_dir):
                for f in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, f))
                os.rmdir(self.temp_dir)
        except Exception as e:
            print("Cleanup warning:", e)


# =========================
# S3 PDF fetch
# =========================
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

def test_local_resume(pdf_path: str, dpi=300, conf=0.15):
    print("🧪 Running local resume test")
    print(f"📄 File: {pdf_path}")

    pdf_bytes = load_local_pdf(pdf_path)

    extractor = LayoutClassExtractor(
        pdf_bytes=pdf_bytes,
        dpi=dpi,
        conf=conf
    )

    results = extractor.extract()

    print(results)

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    # Save output locally (non-Lambda)
    with open(LOCAL_OUTPUT_FILE_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Test completed")
    print(f"📦 Output saved as {LOCAL_OUTPUT_FILE_PATH}")

    return results


# =========================
# Lambda handler
# =========================
def handler(event, context):
    start_time = time.time()

    try:
        print("✅ Lambda invoked")

        req_body = json.loads(event["body"]) if "body" in event else event

        aws_access_key = req_body["aws_access_key"]
        aws_secret_key = req_body["aws_secret_key"]
        pdf_url = req_body["pdf_url"]

        confidence_threshold = req_body.get("confidence_threshold", 0.15)
        dpi = req_body.get("dpi", 300)

        pdf_bytes = fetch_pdf_from_s3(
            pdf_url,
            aws_access_key,
            aws_secret_key
        )

        extractor = LayoutClassExtractor(
            pdf_bytes=pdf_bytes,
            conf=confidence_threshold,
            dpi=dpi,
        )

        results = extractor.extract()

        print("✅ Extraction completed")

        print(f"📋 Results: \n{results}")

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
    # Change path to your local resume PDF
    # LOCAL_PDF_PATH = "TANVI GAWALI CV.pdf"
    # LOCAL_PDF_PATH = "Ujjwal Tyagi.pdf"
    LOCAL_PDF_PATH = "Puunita Chaturvedi.pdf"

    test_local_resume(
        pdf_path=LOCAL_PDF_PATH,
        dpi=300,
        conf=0.15
    )