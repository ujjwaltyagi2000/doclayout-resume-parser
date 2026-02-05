"""
Built over ordered_classes_within_sections.py.

This module: 
    1. imports get_headings() function from resuscan_getheadings.py (resuscan code)
    2. identifies font sizes of the headings from resuscan
    3. runs yolo over the document and maps extracted headings from get_headings() to their corresponding bounding boxes
    4. extracts headings (class 7) from yolo output
    5. find font sizes of yolo extracted headings
    6. find headers with the same font size as resuscan headings
    5. builds sections from extracted headings by sorting the bounding boxes along x-axis, followed by y-axis
    6. this enables multi-column resume parsing
    7. within section headers, text is now in the same ordered (coupled with their class names) as in the original resume

Status: In Progress 🚀
"""

from doclayout_yolo import YOLOv10
from resuscan_fonts import get_headings, get_heading_font_sizes  # Added get_heading_font_sizes
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
        self.pdf_bytes = pdf_bytes
        self.dpi = dpi
        self.conf = conf

        # Lambda-safe temp dir
        self.temp_dir = os.path.join("/tmp", str(uuid.uuid4()))
        # self.temp_dir = os.path.join(os.getcwd(), "saved_cv_pages")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.model = MODEL
        self.class_names = CLASS_NAMES

        # -------------------------
        # Step 1: Get headings
        # -------------------------
        (
            _,
            sub_headings,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            word_to_size
        ) = get_headings(self.pdf_bytes)

        self.sub_headings = {
            self._normalize(h) for h in sub_headings
        }
        self.word_to_size = word_to_size
        
        print(f"🔎Sub Headings: {sub_headings}")
        
        # Get the font sizes for headings
        heading_font_map = get_heading_font_sizes(sub_headings, self.word_to_size)
        print("📃 Heading font map:", heading_font_map)
        
        heading_font_sizes = list(set(heading_font_map.values()))
        print("📃 Heading font sizes:", heading_font_sizes)
        
        # Store for later use
        self.heading_font_map = heading_font_map
        self.target_font_sizes = heading_font_sizes

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
    
    def _get_class_7_content(self):
        """
        Extract class 7 headers with their font sizes.
        Only includes headers whose font sizes match the target_font_sizes from get_headings().
        Returns a dict where each header maps to its font size info.
        """
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        headers_with_fonts = {}
        
        print(f"\n🎯 Target font sizes for filtering: {self.target_font_sizes}")
        print(f"🔍 Processing {sum(1 for b in self.detected_blocks if b['class_id'] == 7)} class 7 headers...\n")
        
        matched_count = 0
        unmatched_count = 0
        
        for block in self.detected_blocks:
            if block["class_id"] == 7:
                text = block["text"]
                page_idx = block["page"]
                page = doc[page_idx]
                
                # Get the exact bounding box from the block
                rect = fitz.Rect(
                    block["x"],
                    block["y"],
                    block.get("x1", block["x"] + 100),  # Use stored x1 or approximate
                    block.get("y1", block["y"] + 50)    # Use stored y1 or approximate
                )
                
                # Extract text with font information using dict format
                blocks_dict = page.get_text("dict", clip=rect)
                font_size = None
                font_name = None
                
                # Parse through the text blocks to find font info
                for text_block in blocks_dict.get("blocks", []):
                    if "lines" in text_block:
                        for line in text_block["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip():
                                    font_size = round(span["size"], 2)
                                    font_name = span.get("font", "Unknown")
                                    break
                            if font_size:
                                break
                    if font_size:
                        break
                
                # Check if font size matches (round to nearest integer for comparison)
                if font_size is not None:
                    rounded_font_size = round(font_size)
                    matches_target = rounded_font_size in self.target_font_sizes
                    
                    # Only include headers that match the target font sizes
                    if matches_target:
                        headers_with_fonts[text] = {
                            "font_size": font_size,
                            "font_size_rounded": rounded_font_size,
                            "font_name": font_name if font_name else None,
                            "page": page_idx + 1,  # 1-indexed for readability
                            "matched_target": True
                        }
                        matched_count += 1
                        print(f"✅ MATCHED: '{text}' | Font: {font_size} (rounded: {rounded_font_size}) | Page: {page_idx + 1}")
                    else:
                        unmatched_count += 1
                        print(f"❌ SKIPPED: '{text}' | Font: {font_size} (rounded: {rounded_font_size}) | Page: {page_idx + 1} | Not in {self.target_font_sizes}")
                else:
                    unmatched_count += 1
                    print(f"⚠️  NO FONT: '{text}' | Could not extract font size | Page: {page_idx + 1}")
        
        doc.close()
        print(f"\n📊 SUMMARY: {matched_count} matched ✅ | {unmatched_count} skipped ❌ | {matched_count + unmatched_count} total\n")
        return headers_with_fonts


    def build_final_output(self, sections):
        final_output = {
            "meta": {
                "dpi": self.dpi,
                "confidence": self.conf,
                "total_pages": len(self.pages_info),
                "target_heading_font_sizes": self.target_font_sizes,  # Added
                "heading_font_map": self.heading_font_map  # Added
            },

            "seperator1": """

--------------------------------------------------------------------------------------------------------------------

""",

            # 1️⃣ Sub-headings from get_headings
            "sub_headings": sorted(set(self.sub_headings)),

            "seperator2": """

--------------------------------------------------------------------------------------------------------------------

""",
            # # 2️⃣ Class-wise extracted content
            # "classes_and_content": self._build_class_wise_content(),

            # 2️⃣ YOLO detected section headers (class 7 only) WITH FONT SIZES - FILTERED BY TARGET SIZES
            "section_headers_yolo": self._get_class_7_content(),

            "seperator3": """

--------------------------------------------------------------------------------------------------------------------

""",
            # 3️⃣ Sectioned content (your existing output)
            "sections_by_header": sections,

            "seperator4": """

--------------------------------------------------------------------------------------------------------------------

""",
            # 4️⃣ Entire resume text line-by-line
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