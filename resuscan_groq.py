from font_based_line_grouping import filter_body_content
from groq import Groq
import os

IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
DEFAULT_OUTPUT_DIR = "/tmp" if IS_LAMBDA else os.getcwd()


# Load .env only if running locally
if os.getenv("AWS_LAMBDA_FUNCTION_NAME") is None:
    from dotenv import load_dotenv
    load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def filter_headers_with_groq(headers):

    prompt = f"""
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


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    cleaned_headers = response.choices[0].message.content

    return cleaned_headers

def load_local_pdf(pdf_path: str) -> bytes:
    with open(pdf_path, "rb") as f:
        return f.read()

if __name__ == "__main__":

    pdf_file_path = "Puunita Chaturvedi.pdf"
    # pdf_file_path = "resume.pdf"
    # pdf_file_path = "Ujjwal Tyagi.pdf"
    # pdf_file_path = "TANVI GAWALI CV.pdf"
    # pdf_file_path = "Megha resume.pdf"
    
    pdf_file = load_local_pdf(pdf_file_path)

    lines = filter_body_content(pdf_file)
    
    # Print line by line
    # print(lines)

    # Remove y_position (can try to include as well in an example)
    # filtered_resume_text = [
    #     {k: v for k, v in item.items() if k != 'y_position'}
    #     for item in lines
    # ]

    filtered_resume_text = lines

    print(filtered_resume_text)

    # Feeding filtered text to Llama

    cleaned_headers = filter_headers_with_groq(filtered_resume_text)
    
    print("\n✅ Cleaned Resume Headers from LLaMA:")
    print(cleaned_headers)