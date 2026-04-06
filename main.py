from parser.pdf_parser import load_local_pdf, fetch_pdf_from_s3
from parser.yolo_parser import LayoutParser
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
    pdf_path = "Puunita Chaturvedi.pdf"
    pdf_bytes = load_local_pdf(pdf_path)
    results = process_resume(pdf_bytes)
    print(results)