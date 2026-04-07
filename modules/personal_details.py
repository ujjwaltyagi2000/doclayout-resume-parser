from io import BytesIO
from PIL import Image
import phonenumbers
import pdfplumber
import fitz
import cv2
import re
import io

def get_phones(text):
    try:
        print("✅ Inside get_phones function")
        phone_numbers = []
        phone_numbers1 = []
        phone_numbers2 = []
        for match in phonenumbers.PhoneNumberMatcher(text, "ZZ"):
        
            phone_number = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
        
            phone_numbers.append(phone_number)
        
    

        if len(phone_numbers) == 0:
            matches = re.findall(r'\b[5-9|0]\d{9,11}\b', text)
            for i in matches:
                phone_numbers1.append(i)   



        if len(phone_numbers) == 0 and len(phone_numbers1) == 0:
            regex = re.findall(r'(?:[-+() ]*\d){10,13}', text)

            for i in regex:
                phone_numbers2.append(i)

        phonenumbers_finderSet = set(phone_numbers)
        phonenumbers_finderSet1 = set(phone_numbers1)
        phonenumbers_finderSet2 = set(phone_numbers2)

    
        phone_all = phonenumbers_finderSet.union(phonenumbers_finderSet1)
        phone_all1 = phone_all.union(phonenumbers_finderSet2)
    
        return phonenumbers_finderSet,phonenumbers_finderSet1,phonenumbers_finderSet2,list(phone_all1)
    except Exception as e:
        phone = set()
        phone1 = set()
        phone2 = set()
        phone_all1=set()
        print("get_phones ",e)
        return list(phone),list(phone1),list(phone2),list(phone_all1)

def get_Phones(raw_data):

    potential_names = re.findall(r"\+\d{2}(?: |\-)?\d{1}(?: |\-)?\d{4}(?: |\-)?\d{4}", raw_data)
    
    return potential_names

def get_emails(text):
    try:
        email_finder = re.findall("([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)",text)

        email_finderSet = set(email_finder)
        
        return email_finderSet
    except Exception as e:
        print("get_emails ",e)
        email=set()
        return list(email)
    
def get_url(pdf_file):
    try: 

        urls = []

        url_flag=0
        linkedIN_flag=0

        with pdfplumber.open(BytesIO(pdf_file)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                for word in text.split():
                
                    if 'linkedin.com' in word.lower() or 'linkedin/' in word.lower(): 
                        linkedIN_flag = 1 
                        urls.append(word)
                    if 'github.com' in word.lower() or 'github.io' in word.lower():
                        
                        urls.append(word)
                        url_flag=1

        url_set = set(urls) 
        url1 = list(url_set)   
        return url1,linkedIN_flag,url_flag 
    except Exception as e:
        url1 = []
        flag = 0
        print("get_url ",e)
        return url1,flag,0
    
def check_Images(pdf_file):
    try: 
    # open the file
        pdf_file = fitz.open(stream=io.BytesIO(pdf_file))
        images_found = False

        # iterate over PDF pages
        for page_index in range(len(pdf_file)):
            # get the page itself
            page = pdf_file[page_index]
            # get image list
            image_list = page.get_images()
            # printing number of images found in this page
            if image_list:
                images_found = True
                # print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        
                for image_index, img in enumerate(image_list, start=1):
                    # get the XREF of the image
            
                    xref = img[0]
            
            
                    # extract the image bytes
                    base_image = pdf_file.extract_image(xref)
            
                    image_bytes = base_image["image"]
            
                    # get the image extension
                    image_ext = base_image["ext"]
            
                    # load it to PIL
                    image = Image.open(io.BytesIO(image_bytes))
            
                    # images = f"image{page_index+1}_{image_index}.{image_ext}"
                    image_path = f"/tmp/image{page_index+1}_{image_index}.{image_ext}"
                    # save it to local disk
                    image.save(open(image_path, "wb"))
                

            

                    # Load the cascade
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    # Read the input image
                    img1 = cv2.imread(image_path)
                    print("this is img1 ---->",img1)
                    # Convert into grayscale
                    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    
                    if len(faces) == 0:
                        return("Human Image Not Detected")
                    else:
                        return("Human Image Detected")

        if not images_found:
            return("No images found in the PDF file.")
            
    except Exception as e:
        print("*************images*************")
        print("images function")
        print(str(e))
        print("error in images found")
        return("No images found in the PDF file.")
