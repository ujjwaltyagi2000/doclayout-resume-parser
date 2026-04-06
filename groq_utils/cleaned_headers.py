from urllib.parse import urlparse
from groq import Groq
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