import fitz
import cv2
import os
import json
from doclayout_yolo import YOLOv10
import time


class LayoutClassExtractor:
    def __init__(
        self,
        pdf_path,
        out_path,
        model_path="doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt",
        dpi=300,
        conf=0.15
    ):
        self.pdf_path = pdf_path
        self.out_path = out_path
        self.dpi = dpi
        self.conf = conf

        os.makedirs(out_path, exist_ok=True)
        os.makedirs("temp", exist_ok=True)

        self.model = YOLOv10(model_path)
        self.class_names = self.model.names  # id -> name
        self.class_texts = {
            name: {"class_id": cid, "texts": []}
            for cid, name in self.class_names.items()
        }

        self.pages_info = self.pdf_to_images()

    def pdf_to_images(self):
        doc = fitz.open(self.pdf_path)
        pages = []

        for i, page in enumerate(doc):
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_path = f"temp/page_{i+1}.png"
            pix.save(img_path)

            pages.append(
                (img_path, pix.width, pix.height, page.rect.width, page.rect.height)
            )

        doc.close()
        return pages

    def pixel_to_pdf_rect(self, box, img_w, img_h, pdf_w, pdf_h):
        x1, y1, x2, y2 = box
        return fitz.Rect(
            x1 * pdf_w / img_w,
            y1 * pdf_h / img_h,
            x2 * pdf_w / img_w,
            y2 * pdf_h / img_h,
        )

    def extract(self):
        doc = fitz.open(self.pdf_path)

        for page_idx, (img_path, img_w, img_h, pdf_w, pdf_h) in enumerate(self.pages_info):
            results = self.model.predict(
                source=img_path,
                imgsz=1024,
                conf=self.conf,
                device="cpu"
            )

            page = doc[page_idx]

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls)
                    class_name = self.class_names[class_id]

                    rect = self.pixel_to_pdf_rect(
                        box.xyxy.cpu().numpy()[0],
                        img_w, img_h, pdf_w, pdf_h
                    )

                    text = page.get_text("text", clip=rect).strip()
                    if text:
                        self.class_texts[class_name]["texts"].append(text)

        doc.close()
        self.cleanup()
        self.save_json()

    def save_json(self):
        output_file = os.path.join(self.out_path, "extracted_classes.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.class_texts, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {output_file}")

    def cleanup(self):
        for f in os.listdir("temp"):
            os.remove(os.path.join("temp", f))
        os.rmdir("temp")


if __name__ == "__main__":

    start_time = time.time()
    extractor = LayoutClassExtractor(
        pdf_path="resume/Siddhant_Jha_Resume.pdf",
        out_path="output",
        conf=0.15
    )
    extractor.extract()
    print(f"⌚ Time taken: {time.time() - start_time} seconds")
