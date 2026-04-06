import fitz
import os
import uuid
from doclayout_yolo import YOLOv10

MODEL_PATH = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"

class LayoutParser:
    def __init__(self, model_path=MODEL_PATH, dpi=300, conf=0.15):
        self.model = YOLOv10(model_path)
        self.class_names = self.model.names
        self.dpi = dpi
        self.conf = conf

        self.temp_dir = os.path.join("/tmp", str(uuid.uuid4()))
        os.makedirs(self.temp_dir, exist_ok=True)

    # -------------------------
    # PDF → images
    # -------------------------
    def _pdf_to_images(self, pdf_bytes):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []

        for i, page in enumerate(doc):
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            path = os.path.join(self.temp_dir, f"page_{i}.png")
            pix.save(path)

            pages.append((path, pix.width, pix.height, page.rect.width, page.rect.height))

        doc.close()
        return pages

    # -------------------------
    # Convert YOLO box → PDF rect
    # -------------------------
    @staticmethod
    def _pixel_to_pdf_rect(box, img_w, img_h, pdf_w, pdf_h):
        x1, y1, x2, y2 = box
        return fitz.Rect(
            x1 * pdf_w / img_w,
            y1 * pdf_h / img_h,
            x2 * pdf_w / img_w,
            y2 * pdf_h / img_h,
        )

    # -------------------------
    # MAIN: Extract layout blocks
    # -------------------------
    def parse(self, pdf_bytes):
        pages_info = self._pdf_to_images(pdf_bytes)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        blocks = []

        for page_idx, (img_path, img_w, img_h, pdf_w, pdf_h) in enumerate(pages_info):

            results = self.model.predict(
                source=img_path,
                imgsz=1024,
                conf=self.conf,
                device="cpu"
            )

            page = doc[page_idx]

            for r in results:
                for box in r.boxes:
                    coords = box.xyxy.cpu().numpy()[0]

                    rect = self._pixel_to_pdf_rect(coords, img_w, img_h, pdf_w, pdf_h)
                    text = page.get_textbox(rect).strip()

                    if not text:
                        continue

                    blocks.append({
                        "page": page_idx,
                        "x": rect.x0,
                        "y": rect.y0,
                        "x1": rect.x1,
                        "y1": rect.y1,
                        "class_id": int(box.cls),
                        "class_name": self.class_names[int(box.cls)],
                        "text": text
                    })

        doc.close()
        self._cleanup()

        # 🔥 Sorting (important for multi-column)
        blocks.sort(key=lambda b: (b["page"], b["x"], b["y"]))

        return blocks

    # -------------------------
    # Cleanup temp images
    # -------------------------
    def _cleanup(self):
        try:
            for f in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, f))
            os.rmdir(self.temp_dir)
        except Exception as e:
            print("Cleanup warning:", e)

if __name__ == "__main__":
    # For local testing
    pdf_path = "Puunita Chaturvedi.pdf"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    parser = LayoutParser()
    blocks = parser.parse(pdf_bytes)
    print(f"Extracted {len(blocks)} blocks")
    print(blocks[:5])  # Print first 5 blocks to verify
