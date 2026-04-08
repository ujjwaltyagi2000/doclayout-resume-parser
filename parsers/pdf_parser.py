from urllib.parse import urlparse
import boto3
import fitz

# Converts PDF URL to bytes for processing
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
    
    response = s3.get_object(Bucket=bucket_name, Key=key)  # full object
    pdf_bytes = response["Body"].read() 

    return pdf_bytes, response 

# converts local PDF to bytes for processing
def load_local_pdf(pdf_path: str) -> bytes:
    with open(pdf_path, "rb") as f:
        return f.read()

def extract_full_text(pdf_bytes: bytes) -> str:
    text = ""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text("text") + "\n"

    doc.close()
    return text.strip()
    
if __name__ == "__main__":

    # For local PDF
    pdf_path = "Puunita Chaturvedi.pdf"
    pdf_bytes = load_local_pdf(pdf_path)
    print(f"Loaded {len(pdf_bytes)} bytes from {pdf_path}")
    print(pdf_bytes[:100])  # Print first 100 bytes to verify")

    # FOR S3 PDF
    pdf_bytes = fetch_pdf_from_s3(
        "https://your-bucket.s3.amazonaws.com/path/to/resume.pdf",
        "your_aws_access_key",
        "your_aws_secret_key"
    )
    print(f"Fetched {len(pdf_bytes)} bytes from S3")