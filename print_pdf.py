import fitz  

def read_pdf_line_by_line(pdf_path):
    doc = fitz.open(pdf_path)

    all_lines = []  # <-- store everything

    for page_number, page in enumerate(doc, start=1):
        print(f"\n--- Page {page_number} ---")

        text = page.get_text()
        lines = text.split("\n")

        for line in lines:
            print(line)

        all_lines.extend(lines)  # <-- accumulate

    doc.close()
    return all_lines


if __name__ == "__main__":

    file_name = "Vinay_P_12042026 (1).pdf"
    DATA_DIR = "documents"
    pdf_file_path = f"{DATA_DIR}/{file_name}"  

    lines = read_pdf_line_by_line(pdf_file_path)

    with open("resume_output_fitz.txt", "w") as f:
        for line in lines:
            f.write(line + "\n")