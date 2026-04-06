
from groq_utils.build_sections_with_groq_headings import *

def test_local_resume(pdf_path: str, prompt = cleaned_headers_prompt_template, dpi=300, conf=0.15):
    print("🧪 Running local resume test")
    print(f"📄 File: {pdf_path}")

    pdf_bytes = load_local_pdf(pdf_path)

    extractor = LayoutClassExtractor(
        pdf_bytes=pdf_bytes,
        prompt=cleaned_headers_prompt_template,
        standard_prompt=standard_headings_prompt,
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
        prompt = req_body.get("prompt", cleaned_headers_prompt_template)
        # standard_headings_prompt = req_body.get("shPrompt", standard_headings_prompt)
        standard_prompt = req_body.get("shPrompt", standard_headings_prompt)

        pdf_bytes = fetch_pdf_from_s3(
            pdf_url,
            aws_access_key,
            aws_secret_key
        )

        extractor = LayoutClassExtractor(
            pdf_bytes=pdf_bytes,
            prompt=prompt,
            standard_prompt=standard_prompt,
            conf=confidence_threshold,
            dpi=dpi
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
        prompt=cleaned_headers_prompt_template,
        dpi=300,
        conf=0.15
    )