from resuscan_getheadings import get_headings

def load_local_pdf(pdf_path: str) -> bytes:
    with open(pdf_path, "rb") as f:
        return f.read()

pdf_bytes = load_local_pdf("resume.pdf")

headers = get_headings(pdf_bytes)

print(headers)