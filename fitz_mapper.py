import fitz

import fitz

class PDFSectionMapper:
    def __init__(self, cleaned_headers):
        self.headers = {self._normalize(h): h for h in cleaned_headers}

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.upper().split())

    # -------------------------
    # Extract lines (preserve order)
    # -------------------------
    def _extract_lines(self, pdf_path):
        doc = fitz.open(pdf_path)
        lines = []

        for page_num, page in enumerate(doc):
            text = page.get_text()  # ✅ correct reading order
            page_lines = text.split("\n")

            for line in page_lines:
                if not line.strip():
                    continue

                lines.append({
                    "text": line.strip(),
                    "norm": self._normalize(line),
                    "page": page_num
                })

        doc.close()
        return lines

    # -------------------------
    # Build sections
    # -------------------------
    def build_sections(self, pdf_path):
        lines = self._extract_lines(pdf_path)

        sections = {}
        current_section = None

        for line in lines:
            norm_text = line["norm"]

            # Header match
            if norm_text in self.headers:
                current_section = self.headers[norm_text]
                sections[current_section] = []
                continue

            if current_section:
                sections[current_section].append(line["text"])

        return sections

if __name__ == "__main__":

    DATA_DIR = "documents"
    pdf_file_path = f"{DATA_DIR}/Vinay_P_12042026 (1).pdf"
    
    cleaned_headers = ['CONTACT', 'SKILLS', 'HONOR AND AWARDS', 'EXPERIENCE', 'EDUCATION']

    mapper = PDFSectionMapper(cleaned_headers)
    sections = mapper.build_sections(pdf_file_path)

    for sec, content in sections.items():
        print(f"\n=== {sec} ===")
        print("\n".join(content[:5]))  # preview

    import json
    # save sections as json
    with open("fitz_sections.json", "w") as f:
        f.write(json.dumps(sections, indent=2))