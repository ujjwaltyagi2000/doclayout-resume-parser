import fitz     
import cv2      
import torch    
import time
import os
import json
from doclayout_yolo import YOLOv10

class TextAndHeaderExtractor:
    def __init__(self, pdf_path, out_path, model_path="doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt", dpi=300):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.pdf_path = pdf_path
        self.model_path = model_path
        self.out_path = out_path
        self.dpi = dpi
        self.pages_info = self.pdf_to_image()
        self.yolo_results = None
        self.all_extracted_text = []
        self.all_text_rects = []

    def pdf_to_image(self):
        """
        Converts each page of pdf_path into a PNG.
        Returns a list of tuples [(png_path, img_width, img_height), …].
        """
        doc = fitz.open(self.pdf_path)
        if len(doc) == 0:
            raise ValueError("Empty PDF.")

        output_info = []
        for page_number in range(len(doc)):
            page = doc[page_number]
            mat = fitz.Matrix(self.dpi/72, self.dpi/72)
            pix = page.get_pixmap(matrix=mat)
            if not os.path.exists("temp"):
                os.makedirs("temp")
            png_path = f"temp/{os.path.basename(self.pdf_path)}_page_{page_number+1}.png"
            pix.save(png_path)
            print(f"Saved page {page_number+1} → {png_path}")

            output_info.append((png_path, pix.width, pix.height, page.rect.width, page.rect.height))
            
        doc.close()
        return output_info

    def cleanup_temp_images(self):
        """
        Cleans up temporary images created during processing.
        """
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(temp_dir)
            print("Temporary images cleaned up.")
        else:
            print("No temporary images to clean up.")

    def run_yolo_on_image(self, image_path,plot=True):
        """
        Runs YOLO on a single image_path.
        Returns a list of bounding‐boxes, where each box is [x1, y1, x2, y2]
        in pixel coordinates of the PNG.
        """
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        print(device)
        model = YOLOv10(self.model_path)
        print(model.names)
        results = model.predict(source=image_path, imgsz=1024, conf=0.15, device=device)
        if plot:
            annotated_frame = results[0].plot(pil=True, line_width=1, font_size=20)
            # annotated_frame.save(f"{self.out_path}/out_{os.path.basename(image_path)}")  
            cv2.imwrite(f"{self.out_path}/out_{os.path.basename(image_path)}", annotated_frame)
            print(f"Results saved to {self.out_path}/out_{os.path.basename(image_path)}")
        
        pixel_boxes = []
        text_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls in [5,7,10]:  
                    # print(f"Class: {box.cls}, Confidence: {box.conf:.2f}, Box: {box.xyxy.cpu().numpy().tolist()[0]}")
                    pixel_boxes.append(box.xyxy.cpu().numpy().tolist()[0])
                if box.cls in [3]:
                    text_boxes.append(box.xyxy.cpu().numpy().tolist()[0])
        # print(len(pixel_boxes), "boxes found in", image_path)
        #sort the boxes based on y1 coordinate
        pixel_boxes.sort(key=lambda x: x[1])  # Sort by y1 coordinate
        text_boxes.sort(key=lambda x: x[1])  # Sort by y1 coordinate
        
        # print(pixel_boxes)
        return pixel_boxes, text_boxes


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
        Opens pdf_path, looks at page_number (0-based), and for each fitz.Rect in `rects`,
        returns the PDF's embedded text inside that rectangle.
        """
        doc = fitz.open(self.pdf_path)
        page = doc[page_number]
        texts = []
        for r in rects:
            txt = page.get_text("text", clip=r).strip()
            texts.append(txt)
        doc.close()
        return texts

    def extract_headers_and_texts(self, plot=True,save=True):
        """
        Extracts headers from the PDF by running YOLO on each page and extracting text.
        """
        for page_idx, (png_path, img_w, img_h, pdf_w, pdf_h) in enumerate(self.pages_info):
            # Run YOLO on the PNG
            pixel_boxes,text_boxes = self.run_yolo_on_image(png_path, plot=plot)
            if not pixel_boxes:
                print(f"No bounding boxes found on page {page_idx + 1}. Skipping.")
                continue
            # Convert pixel boxes to PDF rects
            rects = self.pixel_boxes_to_pdf_rects(pixel_boxes, img_w, img_h, pdf_w, pdf_h)
            text_rects = self.pixel_boxes_to_pdf_rects(text_boxes, img_w, img_h, pdf_w, pdf_h)

            # Extract text from the original PDF under those rects
            texts = self.extract_text_from_pdf(page_idx, rects)
            text_rects_texts = self.extract_text_from_pdf(page_idx, text_rects)
            self.all_extracted_text.append(texts)
            self.all_text_rects.append(text_rects_texts)

        
        # Save content and Clean up temporary images
        if save:
            self.save_headers_to_file()
            self.save_text_to_file()
            self.save_to_json()

        self.cleanup_temp_images()
        # print(self.all_extracted_text)
        return self.all_extracted_text, self.all_text_rects

    def get_beautifully_formatted_headers(self):
        """
        Returns the extracted text in a nicely formatted string.
        """
        formatted_text = ""
        for page_idx, texts in enumerate(self.all_extracted_text):
            formatted_text += f"\n--- Page {page_idx+1} ---\n"
            for i, text in enumerate(texts):
                if repr(text).strip() != "":
                    formatted_text += f"Box#{i}: {repr(text)}\n"
                    # formatted_text += "\n"
        return formatted_text.strip()
    
    def get_beautifully_formatted_text(self):
        """
        Returns the extracted text in a nicely formatted string.
        """
        formatted_text = ""
        for page_idx, texts in enumerate(self.all_text_rects):
            formatted_text += f"\n--- Page {page_idx+1} ---\n"
            for i, text in enumerate(texts):
                if repr(text).strip() != "":
                    formatted_text += f"{repr(text)}\n"
                    # formatted_text += "\n"
        return formatted_text.strip()
    
    def save_headers_to_file(self):
        """
        Saves the extracted headers to a text file.
        """
        output_file = f"{self.out_path}/headers_{os.path.basename(self.pdf_path)}.txt"
        with open(output_file, 'w') as f:
            # f.write(self.get_beautifully_formatted_text())
            f.write(self.get_beautifully_formatted_headers())
        print(f"Results saved to {output_file}")
    
    def save_text_to_file(self):
        """
        Saves the extracted headers to a text file.
        """
        output_file = f"{self.out_path}/text_{os.path.basename(self.pdf_path)}.txt"
        with open(output_file, 'w') as f:
            # f.write(self.get_beautifully_formatted_text())
            f.write(self.get_beautifully_formatted_text())
        print(f"Results saved to {output_file}")
    
    def save_to_json(self):
        """
        Saves the extracted headers and text to a JSON file with headers as keys and their content as values.
        """
        # Get full PDF text
        doc = fitz.open(self.pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        # Flatten headers from all pages
        headers = []
        for page_texts in self.all_extracted_text:
            headers.extend([text for text in page_texts if text.strip()])
        
        # Create dictionary mapping headers to their content
        json_data = {}
        
        for i, header in enumerate(headers):
            # Find where this header appears in the full text
            header_index = full_text.find(header)
            
            if header_index != -1:
                # Find the next header to know where this section ends
                if i < len(headers) - 1:
                    next_header = headers[i + 1]
                    next_header_index = full_text.find(next_header, header_index + len(header))
                    if next_header_index != -1:
                        # Extract content between this header and next header
                        content = full_text[header_index + len(header):next_header_index].strip()
                    else:
                        # If next header not found, take remaining text
                        content = full_text[header_index + len(header):].strip()
                else:
                    # This is the last header, take all remaining text
                    content = full_text[header_index + len(header):].strip()
                
                json_data[header] = content
        
        output_file = f"{self.out_path}/extracted_textv2.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to {output_file}")


if __name__ == "__main__":
    start_time = time.time()
    # pdf_path = "resume/66c526ef-0081-40e6-8ebe-5102e2993746Backend Developer_1752129955434.pdf" # Change this to your PDF file path
    # # pdf_path = "Ujjwal Tyagi Resume Sept 2025.pdf" # Change this to your PDF file path
    pdf_path = "resume/Siddhant_Jha_Resume.pdf" # Change this to your PDF file path
    # # pdf_path = "Shrey Dhawan Basic Resume.pdf" # Change this to your PDF file path
    # pdf_path = "Umang Bhola - Resume.pdf" # Change this to your PDF file path
    out_path = "headers_text_output/" # Change this to your output directory
    model_path = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt" # do not change this unless you have a different model

    extractor = TextAndHeaderExtractor(pdf_path, out_path, model_path=model_path)
    extracted_text, texts = extractor.extract_headers_and_texts(plot=True,save=True)

    # # testing all functions
    # result = extractor.pdf_to_image()
    # print(result)

    print("\n\n===========================================================")
    print("Headers:")
    print(extractor.get_beautifully_formatted_headers())
    print("===========================================================\n\n")
    print("Bullet Points:")
    print(extractor.get_beautifully_formatted_text())
    print("=============================================================")
    end_time = time.time()
    print("\n⌚ Time taken:", end_time - start_time, "seconds")