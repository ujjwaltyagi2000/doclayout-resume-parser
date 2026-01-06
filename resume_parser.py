import os
import cv2
import json
import fitz
from doclayout_yolo import YOLOv10
import time

class HeaderExtractor:
    def __init__(self, pdf_path, out_path, model_path="doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt", dpi=300):
        self.pdf_path = pdf_path
        self.model_path = model_path
        self.out_path = out_path
        self.dpi = dpi
        self.pages_info = self.pdf_to_image()
        self.all_extracted_text = []

    def pdf_to_image(self):
        """
        Converts each page of pdf_path into a PNG.
        Returns a list of tuples [(png_path, img_width, img_height, pdf_width, pdf_height), …].
        """
        doc = fitz.open(self.pdf_path)
        if len(doc) == 0:
            raise ValueError("Empty PDF.")

        output_info = []
        for page_number in range(len(doc)):
            page = doc[page_number]
            mat = fitz.Matrix(self.dpi/72, self.dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            if not os.path.exists("temp_images"):
                os.makedirs("temp_images")
            
            png_path = f"temp_images/{os.path.basename(self.pdf_path)}_page_{page_number+1}.png"
            pix.save(png_path)
            print(f"Saved page {page_number+1} → {png_path}")

            output_info.append((png_path, pix.width, pix.height, page.rect.width, page.rect.height))
            
        doc.close()
        return output_info

    def cleanup_temp_images(self):
        """
        Cleans up temporary images created during processing.
        """
        temp_dir = "temp_images"
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    print("Removing file:", file_path)
                    # os.remove(file_path)
            # os.rmdir(temp_dir)
            print("Temporary images cleaned up.")

    def run_yolo_on_image(self, image_path, plot=True):
        """
        Runs YOLO on a single image_path.
        Returns a list of bounding boxes for headers (class 5, 7, 10).
        """
        # Use CPU - no CUDA required
        model = YOLOv10(self.model_path)
        # results = model.predict(source=image_path, imgsz=1024, conf=0.10, device='cpu')
        results = model.predict(source=image_path, imgsz=1024, conf=0.05, device='cpu')
        
        if plot:
            annotated_frame = results[0].plot(pil=True, line_width=1, font_size=20)
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
            output_path = f"{self.out_path}/out_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated_frame)
            print(f"Results saved to {output_path}")
        
        pixel_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Classes 5, 7, 10 typically represent titles/headers in document layout
                # if int(box.cls) in [5, 7, 10]:  
                # Try all text-related classes
                if int(box.cls) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    pixel_boxes.append(box.xyxy.cpu().numpy().tolist()[0])
        
        # Sort boxes by y-coordinate (top to bottom)
        pixel_boxes.sort(key=lambda x: x[1])
        print(f"Found {len(pixel_boxes)} header boxes in {image_path}")
        return pixel_boxes

    def pixel_boxes_to_pdf_rects(self, pixel_boxes, img_w, img_h, pdf_w, pdf_h):
        """
        Converts pixel boxes to fitz.Rect objects in PDF points.
        """
        scale_x = pdf_w / img_w
        scale_y = pdf_h / img_h

        rects = []
        for x1, y1, x2, y2 in pixel_boxes:
            pdf_x1 = x1 * scale_x
            pdf_y1 = y1 * scale_y
            pdf_x2 = x2 * scale_x
            pdf_y2 = y2 * scale_y
            rects.append(fitz.Rect(pdf_x1, pdf_y1, pdf_x2, pdf_y2))
        return rects

    def extract_text_from_pdf(self, page_number, rects):
        """
        Extracts text from PDF at specified rectangles.
        """
        doc = fitz.open(self.pdf_path)
        page = doc[page_number]
        texts = []
        for r in rects:
            txt = page.get_text("text", clip=r).strip()
            texts.append(txt)
        doc.close()
        return texts

    def extract_headers(self, plot=True, save=True):
        """
        Extracts headers from the PDF by running YOLO on each page.
        """
        for page_idx, (png_path, img_w, img_h, pdf_w, pdf_h) in enumerate(self.pages_info):
            pixel_boxes = self.run_yolo_on_image(png_path, plot=plot)
            
            if not pixel_boxes:
                print(f"No header boxes found on page {page_idx + 1}. Skipping.")
                continue
            
            rects = self.pixel_boxes_to_pdf_rects(pixel_boxes, img_w, img_h, pdf_w, pdf_h)
            texts = self.extract_text_from_pdf(page_idx, rects)
            self.all_extracted_text.append(texts)
        
        if save:
            self.save_results_to_file()

        self.cleanup_temp_images()
        return self.all_extracted_text

    def get_beautifully_formatted_text(self):
        """
        Returns the extracted text in a nicely formatted string.
        """
        formatted_text = ""
        for page_idx, texts in enumerate(self.all_extracted_text):
            formatted_text += f"\n--- Page {page_idx+1} ---\n"
            for i, text in enumerate(texts):
                if text.strip():
                    formatted_text += f"Box#{i}: {repr(text)}\n"
        return formatted_text.strip()
    
    def save_results_to_file(self):
        """
        Saves the extracted headers to a text file.
        """
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        output_file = f"{self.out_path}/headers_{os.path.basename(self.pdf_path)}.txt"
        with open(output_file, 'w') as f:
            f.write(self.get_beautifully_formatted_text())
        print(f"Results saved to {output_file}")


def extract_text_by_headers(text, headers):
    """
    Splits text into sections based on detected headers.
    """
    header_dict = {}
    current_header = None
    start_index = 0

    # Handle text before the first header
    if headers and headers[0] in text:
        first_header_index = text.find(headers[0])
        if first_header_index > 0:
            before_first_header_text = text[:first_header_index].strip()
            if before_first_header_text:
                header_dict["Introduction"] = before_first_header_text
    
    for header in headers:
        header_index = text.find(header, start_index)
        if header_index != -1:
            if current_header is not None:
                header_dict[current_header] = text[start_index:header_index].strip()
            current_header = header
            start_index = header_index + len(header)

    # Handle the last section
    if current_header is not None:
        header_dict[current_header] = text[start_index:].strip()

    return header_dict


# Main execution
if __name__ == "__main__":
    start_time = time.time()
    
    # Configuration
    # pdf_path = "Naman Birla Resume.pdf"  # Change to your PDF path
    # pdf_path = "Ujjwal Tyagi Resume Sept 2025.pdf"  # Change to your PDF path
    # pdf_path = "Siddhant_Jha_Resume.pdf"  # Change to your PDF path
    pdf_path = "resume/66c526ef-0081-40e6-8ebe-5102e2993746Backend Developer_1752129955434.pdf"  # Change to your PDF path
    out_path = "output"
    model_path = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
    
    # Extract headers
    print("Extracting headers from resume...")
    extractor = HeaderExtractor(pdf_path, out_path, model_path=model_path)
    # extracted_text = extractor.extract_headers(plot=False, save=True)
    extracted_text = extractor.extract_headers(plot=True, save=True)
    
    # Filter for uppercase headers (typical resume section headers)
    headers = [text for sublist in extracted_text for text in sublist 
               if text.isupper() and len(text) > 1]
    
    print(f"\nDetected headers: {headers}")
    
    # Extract full text from PDF
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    # Split text by headers
    header_dict = extract_text_by_headers(full_text, headers)
    
    print("\n" + "="*50)
    print("EXTRACTED SECTIONS:")
    print("="*50)
    for header, content in header_dict.items():
        print(f"\n[{header}]")
        print(content[:200] + "..." if len(content) > 200 else content)
    
    print(f"\nTime taken: {time.time() - start_time:.2f} seconds")