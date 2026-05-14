"""

Test script to print resume text lines with their y coordinates and font information using pdfminer

"""

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar

def read_pdf_text_font_y(pdf_path):

    all_lines = []

    for page_number, page_layout in enumerate(extract_pages(pdf_path), start=1):

        print(f"\n--- Page {page_number} ---")

        for element in page_layout:

            if isinstance(element, LTTextContainer):

                for text_line in element:

                    if isinstance(text_line, LTTextLine):

                        line_text = text_line.get_text().strip()

                        if not line_text:
                            continue

                        # y coordinate
                        y0 = text_line.y0

                        # Extract first font found in line
                        font_name = "Unknown"

                        for char in text_line:

                            if isinstance(char, LTChar):
                                font_name = char.fontname
                                break

                        print(
                            f"y={y0:.2f} | "
                            # f"font={font_name} | "
                            f"text={line_text}"
                        )

                        all_lines.append({
                            "page": page_number,
                            "y": y0,
                            # "font": font_name,
                            "text": line_text
                        })

    return all_lines


if __name__ == "__main__":

    file_name = "Devu Siva Durga Sai.pdf"

    DATA_DIR = "Check PDFs"

    pdf_file_path = f"{DATA_DIR}/{file_name}"

    lines = read_pdf_text_font_y(pdf_file_path)

    with open("resume_output_pdfminer.txt", "w") as f:

        for item in lines:

            f.write(
                f"page={item['page']} "
                f"y={item['y']:.2f} "
                # f"font={item['font']} "
                f"text={item['text']}\n"
            )