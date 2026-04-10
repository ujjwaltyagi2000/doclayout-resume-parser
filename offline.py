#-- Import All libraries
    
import json
import boto3
import io
from io import BytesIO
# import unicodedata
# import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTAnno, LAParams
# from matplotlib.colors import rgb2hex
from pdfminer.high_level import extract_text
import os
import webcolors
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
import nltk
import pandas as pd
import pdfplumber
import re
# import sys
from collections import Counter
# import language_tool_python
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import fitz # PyMuPDF

from PIL import Image
import cv2
import tabula
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import en_core_web_lg
nlp1 = en_core_web_lg.load()
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

# from PyPDF2 import PdfReader

import phonenumbers
from phonenumbers import geocoder,carrier
from urlextract import URLExtract
import requests
import urllib.request
from spacy.matcher import Matcher
import spacy

# -- In AWS Lambda functions, the event parameter typically contains information about the --
#  -- triggering event, and the context parameter provides information about the runtime environment.
def handler(event, context):
    try:
        #  extract the ''body' property from the event dictionary and assigns it to the variable req_body
        print("✅ Inside handler function")
        req_body = event['body']
        req_data = json.loads(req_body)
        key = req_data['key']
        experience=req_data['experience']
        file_name = req_data['keyName']
        root_url = req_data.get('rootUrl', 'https://mployee-api.padhakku.com')
        aws_access_key = req_data.get('awsAccessKey')
        aws_secret_key = req_data.get('awsSecretKey')
        aws_region = req_data.get('awsRegion')
        aws_s3_bucket_name = req_data.get('s3BucketName')
        # print(key)
        # print("keyyyyy")
        # print(experience)
        # print("experience")
        # print(file_name)
        print(f"\n🔑 Key: {key}\n🚀 Experience: {experience}\n📁 File Name: {file_name}")
        nltk.data.path.append("nltk_data")

        name, extension = os.path.splitext(file_name)
        file_name = name.replace('_', ' ')
        
        print(file_name)

        linkedInUrl = ""

        print("✅ Payload received")

        # def pdfText(pdf_file):
        #     try:
                
        #         print("✅ Inside pdfText function")

        #         output_string = BytesIO()

        #         resource_manager = PDFResourceManager()
        #         laparams = LAParams()
        #         print("aa1")
        #         device = TextConverter(resource_manager, output_string, laparams=laparams)
        #         interpreter = PDFPageInterpreter(resource_manager, device)
        #         print("aa2")
        #         for page in PDFPage.get_pages(BytesIO(pdf_file)):
        #             interpreter.process_page(page)
        #         print("aa3")
        #         text = output_string.getvalue().decode()
                
        #         device.close()
        #         output_string.close()
        #         print("text returns")
        #         return text
        #     except Exception as e:
        #         print(e)
        #         return("") 

        def pdfText(pdf_file):
            """
            Extracts text from a PDF using PyMuPDF (fitz).

            Args:
                pdf_file (bytes): The PDF file content as bytes.

            Returns:
                str: The extracted text from the PDF.
            """
            try:
                print("✅ Inside pdfText function (using fitz)")

                # Open PDF from bytes
                # pdf_stream = BytesIO(pdf_file)
                # doc = fitz.open(stream=pdf_stream, filetype="pdf")
                
                doc = fitz.open(pdf_file)

                text = ""
                for i, page in enumerate(doc):
                    print(f"🔹 Extracting text from page {i+1}")
                    text += page.get_text("text")  # You can also use "blocks" or "dict" for structured output
                
                doc.close()
                print("✅ Text extraction complete")
                return text.strip()

            except Exception as e:
                print(f"❌ Error in pdfText: {e}")
                return ""



        #Extracting maximum Size of Words in the File
        def get_maxSize_words(pdf_file):
            """
            Extracts the maximum size of words in the given PDF file
            :param pdf_file: The PDF file to extract the maximum size of words from.
            :type pdf_file: bytes
            :return: A tuple containing the maximum size of words and a content size flag. The content size flag is 1 if the maximum size of words is 10, 11 or 12, and 0 otherwise.
            :rtype: tuple[int, int]
            """
            try:
                
                print("✅ Inside get_maxSize_words function")
                word_to_size = {}
                size_to_word = {}


                for page_layout in extract_pages(BytesIO(pdf_file)):
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            for text_object in element:
                                if isinstance(text_object, (LTChar, LTAnno)):
                                    continue  # Skip individual characters and annotations
                                font_size = None
                                current_word = ''
                                for character in text_object:
                                    if isinstance(character, LTChar):
                                        if font_size is None:
                                            font_size = round(character.size)
                                        if character.get_text().isspace():
                                            # End of word
                                            if current_word:
                                                if current_word not in word_to_size:
                                                    word_to_size[current_word] = [font_size]
                                                else:
                                                    word_to_size[current_word].append(font_size)
                                                if font_size not in size_to_word:
                                                    size_to_word[font_size] = [current_word]
                                                else:
                                                    size_to_word[font_size].append(current_word)
                                            current_word = ''
                                            font_size = None
                                        else:
                                            current_word += character.get_text()
                                            font_size = round(character.size)
                                # Check if there is a last word
                                if current_word:
                                    if current_word not in word_to_size:
                                        word_to_size[current_word] = [font_size]
                                    else:
                                        word_to_size[current_word].append(font_size)
                                    if font_size not in size_to_word:
                                        size_to_word[font_size] = [current_word]
                                    else:
                                        size_to_word[font_size].append(current_word)

            
                max_size = None
                max_words = []
                other_sizes = {}

                for size, words in size_to_word.items():
                    if max_size is None or len(words) > len(max_words):
                        max_size = size
                        max_words = words
                

            

                for size, words in size_to_word.items():
                    if size != max_size:
                        other_sizes[size] = words

            
                content_size_flag = 0
                if(max_size==12 or max_size==11 or  max_size==10):
                    content_size_flag=1

                return max_size,content_size_flag
            except Exception as e:
                print("get_maxSize_words ",e)
                max_size = 0
                content_size_flag = 0
                return max_size,content_size_flag

        #Extracting Font Style and Font Size
        def get_font_style_size(pdf_file):
            """
            Function to extract font style and font size from a given pdf file
            Returns a tuple containing a set of font styles, a flag indicating if the font style is standard or not, the number of font styles, a set of font sizes, and the number of font sizes
            """
            print("✅ Inside get_font_style_size function")
            print("styling")
            Extract_Data = []
            Extract_Styles = []
            try:
                for page_layout in extract_pages(BytesIO(pdf_file)):
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            for text_line in element:
                                if isinstance(text_line, LTAnno):
                                    continue  # Skip LTAnno objects
                                if isinstance(text_line, LTChar):
                                    text_line = [text_line]
                                for character in text_line:
                                    if isinstance(character, LTChar):
                                        Font_size = character.size
                                        Font_style=character.fontname
                                        
                                        Extract_Data.append([Font_size, character.get_text()])
                                        if character.get_text() == ' ' :
                                            continue
                                    
                                        Extract_Styles.append(Font_style)
                                        

                font_sizes = set()

                for item in Extract_Data: 
                    font_size = round(item[0], 0)
                    if font_size == 0.0:
                        continue
                    font_sizes.add(font_size)
            
                    

                font_styles = set()
            
                
                for item in Extract_Styles:
            
                    if item == 'SymbolMT':
                        continue 
                    if '+' not in item:
                
                        if '-' in item:
                            font_styles.add(item.split('-')[0])  
                            continue
                        elif ',' in item:
                            font_styles.add(item.split(',')[0])  
                            continue
                        font_styles.add(item)
                        continue
                
                
                    name_parts = item.split("+")
                
                    if name_parts[1] == 'SymbolMT':
                        continue

                    elif len(name_parts) >= 2:
                        
                        if '-' in name_parts[1]:
                
                            style_parts = name_parts[1].split("-")
                
                            font_name = style_parts[0]

                        elif ',' in name_parts[1]:   
                            style_parts = name_parts[1].split(",")
                            font_name = style_parts[0]
                        
                        
                        else:
                            font_name = name_parts[1]    
                    else:
                        continue
                
                    font_styles.add(font_name)
                    
                standard_font_style_flag = 0
                accepted_styles = [ 'Arial','ArialMT','ArialMTPro','ArialUnicodeMS','Calibri','Calibri Light','Cambria','Constantia','Georgia','Helvetica','Lato','Times New Roman','Arial Narrow','ArialNarrow','Book Antiqua','CambriaMath','Constantia','Garamond','Helvetica','Lato','Times New Roman','TimesNewRomanPS','TimesNewRomanPSMT','Verdana','Wingdings'] 
                font_styles_lower = {style.lower() for style in font_styles}
                accepted_styles_lower = [style.lower() for style in accepted_styles]
                standard_font_style_flag=0
                standard_font_style_flag =  1 if font_styles_lower.issubset(accepted_styles_lower) else 0

                multiple_font_style=len(font_styles)
                multiple_font_size = len(font_sizes)
                print("returning............")
                return list(font_styles),standard_font_style_flag,multiple_font_style,list(font_sizes),multiple_font_size
            
            except Exception as e:
                print("get_font_style_size ",str(e))
                font_styles=set()
                standard_font_style_flag=0
                multiple_font_style = 0
                font_sizes = set()
                multiple_font_size = 0
                return font_styles,standard_font_style_flag,multiple_font_style,font_sizes,multiple_font_size


        # Extracting font colors
        def get_font_color(pdf_file):
        
            try:
                print("✅ Inside get_font_color function")
                fontinfo = set()
                for page_layout in extract_pages(BytesIO(pdf_file)):
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):

                            for text_line in element:
                                if isinstance(text_line, LTAnno):
                                    continue  # Skip LTAnno objects
                                if isinstance(text_line, LTChar):
                                    text_line = [text_line]
                                    
                                for character in text_line:
                                    if isinstance(character, LTChar):
                                        font_color = character.graphicstate.ncolor
                                    
                                        if str(type(font_color)) == "<class 'NoneType'>" :
                                        
                                            continue 
                                        
                                        elif isinstance(font_color, int) or  isinstance(font_color, float): 
                                            if font_color == 0: 
                                                hex_value = "#000000"
                                                fontinfo.add(hex_value)

                                        elif (len(font_color) == 3 or len(font_color) == 4):
                                        
                                            r = int(font_color[0] * 255) & 0xFF
                                        
                                            g = int(font_color[1] * 255) & 0xFF
                                            
                                            b = int(font_color[2] * 255) & 0xFF
                                            
                                            hex_value = '#{:02x}{:02x}{:02x}'.format(r, g, b)
                                        
                                            fontinfo.add(hex_value)

                def convert_rgb_to_names(rgb_tuple):
                    css3_db = CSS3_HEX_TO_NAMES
                    names = []
                    rgb_values = []
                
                    for color_hex, color_name in css3_db.items():
                        names.append(color_name)
                        rgb_values.append(hex_to_rgb(color_hex))

                    kdt_db = KDTree(rgb_values)
                    distance, index = kdt_db.query(rgb_tuple)
                    return names[index]

                font_colors = set()
                for hex_value in fontinfo:
                    rgb_tuple = hex_to_rgb(hex_value)
                
                    color_name = convert_rgb_to_names(rgb_tuple)
                
                    font_colors.add(color_name)

                font_colors_Total = len(font_colors)
            

                
                colors_allow = ["gray","dimgray","darkslateblue","darkcyan","darkblue","black", "blue", "brown", "cadetblue", "chocolate", "cornflowerblue","darkblue", "darkcyan", "darkslateblue", "dimgray", "gray", "indigo", "lightblue", "lightseagreen", "lightskyblue", "lightslategray", "lightsteelblue", "maroon", "midnightblue", "navy", "slateblue", "slategray", "steelblue","darkslategray","white","royalblue","teal","dodgerblue","darkgray","darkolivegreen","saddlebrown","whitesmoke","deepskyblue","lightgray","seagreen","sienna","silver","beige","floralwhite","gainsboro","mediumblue""peru","sandybrown"]

                standard_color_flag = 1 if len(font_colors)>0 else 0
                for color in font_colors:
                    if color not in colors_allow:
                    
                        standard_color_flag = 0
                        break

                black=0
                white=0
                if(font_colors_Total==2 or font_colors_Total==1):
                    for color in font_colors:
                        if color.lower()=="black":
                            black=1
                        elif color.lower()=="white":
                            white=1
                if(black or white):
                    standard_color_flag=0

                print("at font colors return ")
                return list(font_colors),font_colors_Total,standard_color_flag
            
            except Exception as e:
                print("get_font_color ",e)
                font_colors_Total = 0
                standard_color_flag = 0
                font_colors = set()
                return list(font_colors),font_colors_Total,standard_color_flag

        #     # headings
        
        # headings
        # getting actual headings from Resume
        def extract_headings_two(words_to_check_flat):
            try:
                # Read the Excel file into a pandas DataFrame
                print("✅ Inside extract_headings_two function")
                
                # df = pd.read_excel('Headlines_Categories.xlsx')
                df = pd.read_excel('https://s3.ap-south-1.amazonaws.com/mployee.me/keywords_list/Headlines_Categories.xlsx')
                # Initialize an empty dictionary to store the data
                data_dict = {}
                for col_name, col_data in df.items():
                
                    if col_name == df.columns[0]:  # skip the first column
                        continue
                    key = col_data[0]
                    # Get the list of values for the column and convert it to a list of tuples with priorities
                    values = []
                    for row_index, row in col_data.items():
                        
                        # Get the priority from the first column
                        priority =  df.iloc[row_index, 0]
                    
                        # Check if the value is NaN
                        if pd.isna(row):
                            continue
                        values.append((row, priority))
                    # Add the key-value pair to the dictionary
                
                    data_dict[key] = values

                



                
                bigrams_trigrams = list(map(' '.join, nltk.everygrams(words_to_check_flat, 2, 3)))
                # Concatenate the two lists
                combined_list = words_to_check_flat + bigrams_trigrams
                combined_list = [re.sub(':', '', word) for word in combined_list]
                # Create a new map with the keys from data_dict and values from the combined_list
            
                new_map = {}
                for key, values in data_dict.items():
                    matched_words = []
                    for word in combined_list:
                    
                    
                        for value, priority in values:
                            if str(value).lower() == word.lower():
                            
                                matched_words.append((word, priority))
                    if matched_words:
                        new_map[key] = matched_words
                

                subs = list(new_map.keys())
                priority_map={}
                for key,values in new_map.items():
                    min_val =  (2 ** 31) - 1
                    for value,priority in values:
                        if priority < min_val:
                            min_val=priority
                            priority_map[key]=min_val
                # Return the new_map dictionary


                headings_set = set()
                for key,values in new_map.items():
                    for value,priority in values:
                        if priority_map[key]==priority:
                            headings_set.add(value)
                
                lower=[]
                upper=[]
                proper=[]
                
                for heading in headings_set:
                    if heading.islower()==True:
                        lower.append(heading)
                    elif heading.isupper()==True:
                        upper.append(heading)
                        
                    else:
                        proper.append(heading)

                # print("upper--->",upper)
                # print("proper--->",proper)
                # print("lower--->",lower)
                if(len(upper)>0):
                    return upper
                else:
                    return proper
                
                # return headings_set
            except Exception as e:
                print("extract_headings_two ",e)
                headings_set = set()
                return headings_set 
            

        #Matching actual Headings With Standard Headings
        def matchCategories(word_list):
            try: 
                print("✅ Inside matchCategories function")
                # file_path ='Headlines.xlsx'
                file_path = 'https://s3.ap-south-1.amazonaws.com/mployee.me/keywords_list/Headlines.xlsx'
            
                df = pd.read_excel(file_path)

                # Create empty dictionary
                word_dict = {}

                # Loop through each row in the dataframe
                for index, row in df.iterrows():
                    
                    # Get key and value from row
                    key = row[0]
                    value = row[1]

                    # Check if key already exists in dictionary
                    if key in word_dict:
                        # If key already exists, append value to existing list
                        word_dict[key].append(value)
                    else:
                        # If key does not exist, create new list with value
                        word_dict[key] = [value]
            
                NR = set()
                Oth = set()
                ES=set()
                wp=set()
                Oth_db = set()
                word_list = [word.lower() for word in word_list]
                section_map=dict()
                matching_keys = set()
                for key in word_dict:
                    for value in word_dict[key]:
                        
                        if value.lower() in word_list:
                        
                            if key in section_map:
                                section_map[key].append(value)
                            else:
                                section_map[key] = [value]
                            matching_keys.add(key)
                            
                            if(key=="Not Required"):
                                NR.add(value)
                            elif(key=="Work Experience" or key=="Projects"):
                                wp.add(key)
                            elif(key == "Education" or key=="Skills"):
                                ES.add(key)
                            else:
                                Oth.add(value)
                                Oth_db.add(key)
                            # break  # Break out of inner loop once match is found for current key

            
                section_map_count = 0

                for key, value in section_map.items():
                    if len(value) > 1:
                        section_map_count += 1

                return matching_keys,list(NR),list(wp),list(ES),list(Oth),list(Oth_db),section_map,section_map_count
            except Exception as e:
                print("matchCategories ",e)
                matching_keys = set()
                NR = set()
                Oth = set()
                ES=set()
                wp=set()
                Oth_db = set()
                section_map=dict()
                section_map_count = 0
                return matching_keys,list(NR),list(wp),list(ES),list(Oth),list(Oth_db),section_map,section_map_count


        #Matching Found standard headings with expected general headings
        def standard_headingsMatch(word_list):
            try: 
                
                print("✅ Inside standard_headingsMatch function")

                # file_path = 'Standard Headings.xlsx'
                file_path = 'https://s3.ap-south-1.amazonaws.com/mployee.me/keywords_list/Standard+Headings.xlsx'
            
                df = pd.read_excel(file_path)

                
                first_column = df.iloc[:, 0]
                words_array = first_column.to_numpy()
                word_list_lower = [word.lower() for word in word_list]
                standard_match =[]
                for word in words_array:
                    if word.lower() in word_list_lower:
                            standard_match.append(word)
                
                return standard_match,len(standard_match)
            except Exception as e:
                print("standard_headingsMatch ",e)
                standard_match =[]
                return standard_match,len(standard_match)

        #Main Headings code, deciding where to pick headings from 
        def get_headings(pdf_file):
            # scrape and divivde all the words into groups
            try: 
                print("✅ Inside get_headings function")
                word_to_size = {}
                size_to_word = {}


                for page_layout in extract_pages(BytesIO(pdf_file)):
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            for text_object in element:
                                if isinstance(text_object, (LTChar, LTAnno)):
                                    continue  # Skip individual characters and annotations
                                font_size = None
                                current_word = ''
                                for character in text_object:
                                    if isinstance(character, LTChar):
                                        if font_size is None:
                                            font_size = round(character.size)
                                        if character.get_text().isspace():
                                            # End of word
                                            if current_word:
                                                if current_word not in word_to_size:
                                                    word_to_size[current_word] = [font_size]
                                                else:
                                                    word_to_size[current_word].append(font_size)
                                                if font_size not in size_to_word:
                                                    size_to_word[font_size] = [current_word]
                                                else:
                                                    size_to_word[font_size].append(current_word)
                                            current_word = ''
                                            font_size = None
                                        else:
                                            current_word += character.get_text()
                                            font_size = round(character.size)
                                # Check if there is a last word
                                if current_word:
                                    if current_word not in word_to_size:
                                        word_to_size[current_word] = [font_size]
                                    else:
                                        word_to_size[current_word].append(font_size)
                                    if font_size not in size_to_word:
                                        size_to_word[font_size] = [current_word]
                                    else:
                                        size_to_word[font_size].append(current_word)


                max_size = None
                max_words = []
                other_sizes = {}

                for size, words in size_to_word.items():
                    if max_size is None or len(words) > len(max_words):
                        max_size = size
                        max_words = words
                


                for size, words in size_to_word.items():
                    if size != max_size:
                        other_sizes[size] = words

            
                words_to_check=[]
                for words in other_sizes.values():
                    words_to_check.append(words)
            

                words_to_check_flat = [word for sublist in words_to_check for word in sublist]
                total_words = [word for words in size_to_word.values() for word in words]
            

                    
                with pdfplumber.open(BytesIO(pdf_file)) as pdf:
                    word_array = []
                    for page in pdf.pages:
                        clean_text = page.filter(lambda obj: not (obj["object_type"] == "char" and "Bold" in obj["fontname"]))
                        words = clean_text.extract_text().split()
                        for word in words:
                            if all(ord(c) < 128 for c in word):
                                word_array.append(word)
                

                #getting uppercase words
                # Open PDF file
                with pdfplumber.open(BytesIO(pdf_file)) as pdf:
                    
                    # Initialize list to store uppercase words
                    uppercase_words = []
                    
                    
                    # Iterate through each page of the PDF
                    for page in pdf.pages:
                        
                        # Extract text from page and split into words
                        text = page.extract_text()
                        
                        
                        # Loop through each word and check if it is uppercase
                        for word in text.split():
                            if word.isupper() or (word[0].isupper() and "&" in word):
                                uppercase_words.append(word)
                            if word=='&':
                                uppercase_words.append(word)
            

                headings_four=extract_headings_two(words_to_check_flat)
            

                if(len(headings_four)==0):
                    
                    word_array.extend(uppercase_words)
                    headings_three = extract_headings_two(word_array)
                    # headings_two = extract_headings_two(uppercase_words)
                
                    # headings_two = {word.lower() for word in headings_two}
                    # headings_three = {word.lower() for word in headings_three}
                
                    
                    # common_words = headings_three.intersection(headings_two)
                
                    # all_words = list((headings_three - common_words) | (headings_two - common_words) | common_words)
                

                    if(len(headings_three)==0):
                        headings_all = extract_headings_two(total_words)
                    
                    
                        categories_list,nr,wp,es,oth,oth_db,section_map,section_map_count=matchCategories(headings_all)
                        
                        standard_match , standard_match_count = standard_headingsMatch(headings_all)
                        return list(categories_list),list(headings_all),list(nr),list(wp),list(es),list(oth),list(oth_db),section_map,section_map_count,standard_match,standard_match_count
                    else:
                    
                        # print("at uppercase & bold ---->")
                        categories_list,nr,wp,es,oth,oth_db,section_map,section_map_count=matchCategories(headings_three)
                        
                        standard_match , standard_match_count = standard_headingsMatch(headings_three)
                        return list(categories_list),list(headings_three),list(nr),list(wp),list(es),list(oth),list(oth_db),section_map,section_map_count,standard_match,standard_match_count
                else:
                
                    categories_list,nr,wp,es,oth,oth_db,section_map,section_map_count=matchCategories(headings_four)
                    
                    standard_match , standard_match_count = standard_headingsMatch(headings_four)
                    return list(categories_list),list(headings_four),list(nr),list(wp),list(es),list(oth),list(oth_db),section_map,section_map_count,standard_match , standard_match_count 
            except Exception as e:
                print("get_headings ",e)
                return [],[],[],[],[],[],[],[],[],[],[]

        #"Extracting Positive action words from resume text"
        def get_action_words(text):
            try:
                print("✅ Inside get_action_words function")
                file_path = 'ActionWords.xlsx'
                dataframe1 = pd.read_excel(file_path,usecols='A')
                
                SKILLS_DB = dataframe1.values.tolist()
                topics_flat1 = [topic for sublist in SKILLS_DB for topic in sublist]
                topics_flat =  [x.lower() for x in topics_flat1]

                stop_words = set(nltk.corpus.stopwords.words('english'))
                word_tokens = nltk.tokenize.word_tokenize(text)
                print("found stopwords...")
                # remove the stop words
                filtered_tokens = [w for w in word_tokens if w not in stop_words]
            
                # remove the punctuation
                filtered_tokens = [w for w in word_tokens if w.isalpha()]
            
                # generate bigrams and trigrams (such as artificial intelligence)
                bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))   
            

                actionwords = []

                for token in filtered_tokens:
                    
                    if token.lower() in topics_flat:
                    
                        actionwords.append(token.lower())

                for ngram in bigrams_trigrams:
                    if ngram.lower() in topics_flat:
                        actionwords.append(ngram.lower())
                actionwordsSet = set(actionwords)        
                actionwords_total = len(actionwordsSet)
                return list(actionwordsSet),actionwords_total,list(actionwords)
            except Exception as e:
                print("get_action_words ",e) 
                return [],0,[]


    # #     #Detecting frequency of action words either positive or negative
        def frequency_Action_words(actionwords):
            try: 
                print("✅ Inside frequency_Action_words function")
                frequencyList = []
                repeated_frequency={}
                frequency = Counter(actionwords)
                for word in frequency:
                    if frequency[word] >= 3:
                        frequencyList.append(word)
                        repeated_frequency[word]=frequency[word]
                    else:
                        continue
                return frequencyList,len(frequencyList),repeated_frequency
            except Exception as e:
                print("frequency_Action_words ",e) 
                return [],0,{}

        
    #     #Extracting URL from resume
        def get_url(pdf_file):
          try: 
            print("✅ Inside get_url function")
  
            urls = []

            url_flag=0
            linkedIN_flag=0
            linkedIn = None

            with pdfplumber.open(BytesIO(pdf_file)) as pdf:
                for page in pdf.pages:
                 text = page.extract_text()
                 for word in text.split():
                   
                   if 'linkedin.com' in word.lower() or 'linkedin/' in word.lower(): 
                        linkedIN_flag = 1 
                        linkedIn = word
                        urls.append(word)
                   if 'github.com' in word.lower() or 'github.io' in word.lower():
                      
                        urls.append(word)
                        url_flag=1

            url_set = set(urls) 
            url1 = list(url_set)   
            return url1,linkedIN_flag,url_flag,linkedIn
          except Exception as e:
            url1 = []
            flag = 0
            print("get_url ",e)
            return url1,flag,0

    #     # "Detecting negative action words from Resume"  
        def get_negative_action_words(text):
            """
            Extracts negative action words or phrases from a given text.

            This function processes the provided text to identify and return words or 
            multi-word expressions (bigrams or trigrams) that match any entry in a pre-defined 
            list of negative action words (stored in 'Negative Action Words.xlsx').

            The following steps are performed:
            1. Reads a list of negative action words from an Excel file.
            2. Tokenizes the input text and removes stop words and punctuation.
            3. Generates bigrams and trigrams (pairs or triplets of consecutive words).
            4. Compares the cleaned-up tokens (including bigrams and trigrams) with the list of 
            negative action words.
            5. Returns a set of unique matches along with the total count and a list of all 
            matched tokens.

            Args:
                text (str): The input text in which to search for negative action words.

            Returns:
                tuple: A tuple containing:
                    - A list of unique negative action words or phrases found in the text.
                    - The total count of unique negative action words or phrases found.
                    - A list of all matched words/phrases, including duplicates.

            Example:
                text = "The project failed due to poor management and lack of planning."
                negative_words, total_count, all_matches = get_negative_action_words(text)
                print(negative_words)  # Output: ['failed', 'poor management']
                print(total_count)     # Output: 2
                print(all_matches)     # Output: ['failed', 'poor management']
            
            Exceptions:
                - If any error occurs during processing (e.g., file reading, tokenization), 
                an empty list, a count of 0, and an empty list of all matches are returned.
            """
            
            try: 
                print("✅ Inside get_negative_action_words function")
                
                dataframe2 = pd.read_excel('Negative Action Words.xlsx',usecols='A')
                

                SKILLS_DB_negative = dataframe2.values.tolist()
                topics_flat2 = [topic for sublist in SKILLS_DB_negative for topic in sublist]
                topics_flat3 =  [x.lower() for x in topics_flat2]

                stop_words1 = set(nltk.corpus.stopwords.words('english'))
                word_tokens1 = nltk.tokenize.word_tokenize(text)

                # remove the stop words
                filtered_tokens1 = [w for w in word_tokens1 if w not in stop_words1]
            
                # remove the punctuation
                filtered_tokens1 = [w for w in word_tokens1 if w.isalpha()]
            
                # generate bigrams and trigrams (such as artificial intelligence)
                bigrams_trigrams1 = list(map(' '.join, nltk.everygrams(filtered_tokens1, 2, 3)))   
            

                actionwords1 = []

                for token in filtered_tokens1:
                    
                    if token.lower() in topics_flat3:
                    
                        actionwords1.append(token.lower())

                for ngram in bigrams_trigrams1:
                    if ngram.lower() in topics_flat3:
                        actionwords1.append(ngram.lower())
                actionwordsSet_negative = set(actionwords1)       
                actionwords_total_negative = len(actionwordsSet_negative)

                return list(actionwordsSet_negative),actionwords_total_negative,actionwords1
            except Exception as e:
                print("get_negative_action_words ",e)
                actionwordsSet_negative = []
                actionwords_total_negative = 0
                actionwords1 = []
                return actionwordsSet_negative,actionwords_total_negative,actionwords1

    #     # Extract total word count in resume using text
        
        def get_totalWordCount(text):
            try: 
                print("✅ Inside get_totalWordCount function")
                word_list = text.split()
                words_list_final = [s for s in word_list if len(s) != 1 or s.isdigit() or s.isalpha() or s == '|' or s == '/' or s == '-' or s == ':' or s == ';']
                total_word_count = len(words_list_final)
                return total_word_count
            except Exception as e:
                print("get_totalWordCount ",e)
                return 0

    #     # Extracting  emails in resume using text
        def get_emails(text):
            try:
                print("✅ Inside get_emails function")
                email_finder = re.findall("([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)",text)

                email_finderSet = set(email_finder)
                
                return email_finderSet
            except Exception as e:
                print("get_emails ",e)
                email=set()
                return list(email)

    #     # Extracting skills from resume 
        def extract_skills(input_text):

            try:  
                print("✅ Inside extract_skills function")
                file_path = "https://s3.ap-south-1.amazonaws.com/mployee.me/keywords_list/Keywords.xlsx"
                dataframe1 = pd.read_excel(file_path,usecols='A', header=None)
            
                SKILLS_DB = dataframe1.values.tolist()
                topics_flat1 = [topic for sublist in SKILLS_DB for topic in sublist]
                topics_flat =  [x.lower() for x in topics_flat1]
                flag_skill = 0
                stop_words = set(nltk.corpus.stopwords.words('english'))
                word_tokens = nltk.tokenize.word_tokenize(input_text)
            
                # remove the stop words
                filtered_tokens = [w for w in word_tokens if w not in stop_words]
            
                # remove the punctuation
                filtered_tokens = [w for w in word_tokens if w.isalpha()]
            
                # generate bigrams and trigrams (such as artificial intelligence)
                bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
                
                # we create a set to keep the results in.  
                found_skills = set()
                
            
                # we search for each token in our skills database
                for token in filtered_tokens:
                    
                    if token.lower() in topics_flat:
                        found_skills.add(token.lower())

                    
                # we search for each bigram and trigram in our skills database
                for ngram in bigrams_trigrams:
                    if ngram.lower() in topics_flat:
                        found_skills.add(ngram.lower())

                
                Skills_Total = len(found_skills)
            

                return list(found_skills),Skills_Total
            except Exception as e:
                print("Extract_skills ",str(e))
                
                return [],0
                
        def get_pageCount(pdf_file):
            try:  
                
                print("✅ Inside get_pageCount function")
                file_bytes = BytesIO(pdf_file)

                # Count the number of pages in the PDF file
                file_bytes.seek(0)  # Reset the file pointer to the beginning
                parser = PDFParser(file_bytes)
                document = PDFDocument(parser)
                return len(document.catalog['Pages'].resolve()['Kids'])
            except Exception as e:
                print("*********page count*************")
                print(str(e))
                pages_count = 0
                return pages_count



        #Old Pronouns Code
        
        # def detect_personal_pronouns(text):
        #     try: 
        #         # tokenize the text into words
        #         words = nltk.word_tokenize(text)
        #         remove = ['It','itself','Thee']
        #         remove1 = [x.lower() for x in remove]
        #         # use the part-of-speech (POS) tagger to tag each word with its part of speech
        #         tagged_words = nltk.pos_tag(words)
                
        #         # extract personal pronouns from the tagged words
        #         pronouns = set([word for word, tag in tagged_words if tag == 'PRP' or tag == 'PRP$'])
                
                
        #         # return a list of unique pronouns
        #         # print(tagged_words)
        #         pronoun_list = [x.lower() for x in pronouns]
        #         res = [i for i in pronoun_list if i not in remove1]
        #         res_set = set(res)
        #         bullet = get_bullets(text)
        #         flattened_list = flatten_list(bullet)
        #         res_set -= flattened_list
        #         return list(res_set)
            
        #     except Exception as e:
        #         print("***********personal pronouns*************")
        #         print(str(e))
        #         res_set = list()
        #         return res_set



        # def flatten_list(big_list):
        #     flattened_list = []
        #     for element in big_list:
        #         if isinstance(element, list):
        #             flattened_list.extend(element)
        #         else:
        #             flattened_list.append(element)
        #     return set(flattened_list)      

 







        def get_excel_pronouns(text):
            try:
                print("✅ Inside get_excel_pronouns function")
                file_path = 'Pronouns.xlsx'
                dataframe1 = pd.read_excel(file_path,usecols='A')
                
                SKILLS_DB = dataframe1.values.tolist()
                topics_flat1 = [topic for sublist in SKILLS_DB for topic in sublist]
                topics_flat =  [x.lower() for x in topics_flat1]
                print(topics_flat)
                stop_words = set(nltk.corpus.stopwords.words('english'))
                word_tokens = nltk.tokenize.word_tokenize(text)
               
                
                filtered_tokens = [w for w in word_tokens if w not in stop_words]
            
                
                filtered_tokens = [w for w in word_tokens if w.isalpha()]
            
   
                bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))   
            

                pronoun = []

                for token in filtered_tokens:
                    
                    if token.lower() in topics_flat:
                    
                        pronoun.append(token.lower())

                for ngram in bigrams_trigrams:
                    if ngram.lower() in topics_flat:
                        pronoun.append(ngram.lower())
                
                return list(pronoun)
            except Exception as e:
                print("get_pronoun_words ",str(e)) 
                return []

        

        def text_voice(text):
           try: 
                print("✅ Inside text_voice function")
                voice = []
                matcher = Matcher(nlp.vocab)
                doc = nlp(text)
                sents = list(doc.sents)
                # print("Number of Sentences = ", len(sents))
                for sent in doc.sents:
                    for token in sent:
                        print(token.dep_, token.tag_, end=" ")
                    # print()
                passive_rule = [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}]
                matcher.add('Passive', [passive_rule])
                matches = matcher(doc)
                # print(matches)
                for match_id,start,end in matches:
                
                    span = doc[start:end]

                    # print("Matched span:", span.text)

                    voice.append(span.text)
                return voice
             
           except Exception as e:
                print("**************Voice*************************")
                print(str(e))
                return []
           

        
        def get_filler_words(text):
            try:
                print("✅ Inside get_filler_words function")
                file_path = 'Filler Words.xlsx'
                dataframe1 = pd.read_excel(file_path,usecols='A')
                
                SKILLS_DB = dataframe1.values.tolist()
                topics_flat1 = [topic for sublist in SKILLS_DB for topic in sublist]
                topics_flat =  [x.lower() for x in topics_flat1]
                # print(topics_flat)
                stop_words = set(nltk.corpus.stopwords.words('english'))
                word_tokens = nltk.tokenize.word_tokenize(text)
               
                
                filtered_tokens = [w for w in word_tokens if w not in stop_words]
            
                
                filtered_tokens = [w for w in word_tokens if w.isalpha()]
            
   
                bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))   
            

                fillerwords = []

                for token in filtered_tokens:
                    
                    if token.lower() in topics_flat:
                    
                        fillerwords.append(token.lower())

                for ngram in bigrams_trigrams:
                    if ngram.lower() in topics_flat:
                        fillerwords.append(ngram.lower())
                fillerwordsSet = set(fillerwords)        
                fillerwords_total = len(fillerwordsSet)
                return list(fillerwordsSet),fillerwords_total,list(fillerwords)
            except Exception as e:
                print("get_filler_words ",e) 
                return [],0,[]

        
        def get_bold(pdf_file):
            try: 
                print("✅ Inside get_bold function")
                word_to_size = {}
                size_to_word = {}


                for page_layout in extract_pages(BytesIO(pdf_file)):
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            for text_object in element:
                                if isinstance(text_object, (LTChar, LTAnno)):
                                    continue  # Skip individual characters and annotations
                                font_size = None
                                current_word = ''
                                for character in text_object:
                                    if isinstance(character, LTChar):
                                        if font_size is None:
                                            font_size = round(character.size)
                                        if character.get_text().isspace():
                                            # End of word
                                            if current_word:
                                                if current_word not in word_to_size:
                                                    word_to_size[current_word] = [font_size]
                                                else:
                                                    word_to_size[current_word].append(font_size)
                                                if font_size not in size_to_word:
                                                    size_to_word[font_size] = [current_word]
                                                else:
                                                    size_to_word[font_size].append(current_word)
                                            current_word = ''
                                            font_size = None
                                        else:
                                            current_word += character.get_text()
                                            font_size = round(character.size)
                                # Check if there is a last word
                                if current_word:
                                    if current_word not in word_to_size:
                                        word_to_size[current_word] = [font_size]
                                    else:
                                        word_to_size[current_word].append(font_size)
                                    if font_size not in size_to_word:
                                        size_to_word[font_size] = [current_word]
                                    else:
                                        size_to_word[font_size].append(current_word)
            
                # print(size_to_word)
                # print(word_to_size)
                max_size = None
                max_words = []
                

                for size, words in size_to_word.items():
                    if max_size is None or len(words) > len(max_words):
                        max_size = size
                        max_words = words
                
                

                # print(max_words)
                
            
                
                max_string = ' '.join(max_words)
                # print(max_string)

                return max_string 

            except Exception as e:
                print("Bold breaked")
                print(str(e))
                return ""


        def frequent_ngrams(words, n):
            try:
                
                print("✅ Inside frequent_ngrams function")
                # print(words)
            
                ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

        
                ngram_counts = {}
                for gram in ngrams:
                    ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

            
                frequent_ngrams = {
                    gram: count
                    for gram, count in ngram_counts.items()
                    if count >= 2
                }

                return frequent_ngrams
            except Exception as e:
                print(f"frequent_{n}grams", str(e))
                return {}    
            
        




        def get_subWords(raw_data):
          print("✅ Inside get_subWords function")
          potential_names = re.findall(r"\d{1,2}(?:st|nd|rd|th)", raw_data)
          return potential_names
    
        def get_Numbers(raw_data):
            print("✅ Inside get_Numbers function")
            potential_names = re.findall(r"\d+", raw_data)
            return potential_names
    
       
 
        def get_Phones(raw_data):
            print("✅ Inside get_Phones function")
            potential_names = re.findall(r"\+\d{2}(?: |\-)?\d{1}(?: |\-)?\d{4}(?: |\-)?\d{4}", raw_data)
            
            return potential_names
        

        def detect_names(text):
          try:
            print("✅ Inside detect_names function")
            doc = nlp(text)
            names = []
            
            for entity in doc.ents:
                
                if entity.label_ == "PERSON":
                
                    names.append(entity.text)
            return names
          except Exception as e:
              print("******detect_names*****")
              print(str(e))

        

        def get_max_size(pdf_file,txt):
            
            print("✅ Inside get_max_size function")
            word_to_size = {}
            size_to_word = {}


            for page_layout in extract_pages(BytesIO(pdf_file)):
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        for text_object in element:
                            if isinstance(text_object, (LTChar, LTAnno)):
                                continue  # Skip individual characters and annotations
                            font_size = None
                            current_word = ''
                            for character in text_object:
                                
                                if isinstance(character, LTChar):
                                    if font_size is None:
                                        font_size = round(character.size)
                                    if character.get_text().isspace():
                                        # End of word
                                    
                                        if current_word:
                                            
                                            if current_word not in word_to_size:
                                                word_to_size[current_word] = [font_size]
                                            else:
                                                word_to_size[current_word].append(font_size)
                                            if font_size not in size_to_word:
                                                size_to_word[font_size] = [current_word]
                                            else:
                                                size_to_word[font_size].append(current_word)
                                        current_word = ''
                                        font_size = None
                                    else:
                                        current_word += character.get_text()
                                        font_size = round(character.size)
                            # Check if there is a last word
                            if current_word:
                                if current_word not in word_to_size:
                                    word_to_size[current_word] = [font_size]
                                else:
                                    word_to_size[current_word].append(font_size)
                                if font_size not in size_to_word:
                                
                                    size_to_word[font_size] = [current_word]
                                else:
                                    
                                    size_to_word[font_size].append(current_word)
        
            # print(size_to_word)
            max_size = None
            max_words = []
            other_sizes = {}

            for size, words in size_to_word.items():
                if max_size is None or len(words) > len(max_words):
                    max_size = size
                    max_words = words
            


            for size, words in size_to_word.items():
                if size != max_size:
                    other_sizes[size] = words
            
            # print(other_sizes)
            if other_sizes:        
                largest = max(other_sizes.keys())
                bullet = list((set(other_sizes[largest])))
                finalBullet,Bullets_Total,standard_bullet_flag= get_bullets(txt)
                list1 = sorted(bullet)
                list2 = sorted(finalBullet)
                if list1 == list2:
                    del other_sizes[largest]
                    largest = max(other_sizes.keys())
                return other_sizes[largest]
            else:
                return []
        
        # def get_finalise_names(alloutput):

        #     if len(alloutput) < 3:
        #         return ""
        #     else:
        #         return alloutput

        def get_finalise_names(alloutput):
            print("✅ Inside get_finalise_names function")
            if alloutput is None:
                return ""
            elif len(alloutput) < 3:
                return ""
            else:
                return alloutput    

        def match_strings(string1, string2):
            print("✅ Inside match_strings function")
            string1_lower = string1.lower()
            string2_lower = string2.lower()

        
            words1 = string1_lower.split()
            words2 = string2_lower.split()

            for word1 in words1:
                for word2 in words2:
                    if word1 in word2 or word2 in word1:
                        return True
            return False
        


            
        def detect_names_all(pdf_file,raw_data,extra_words):
          try:
             print("✅ Inside detect_names_all function")
             max_size = get_max_size(pdf_file,raw_data)

             if 0 < len(max_size) < 4:  
                    print("From Size")
                    max_size = [word for word in max_size if word not in [extra_word.lower() for extra_word in extra_words]]
                    max_size = [word for word in max_size if word not in [extra_word for extra_word in extra_words]]

                    if len(max_size) == 0:         
                                print("Nothing")
                                print(raw_data.split())
                                raw_data = raw_data.split()[:4]
                                
                                raw_data = [word for word in raw_data if word.lower() not in [extra_word.lower() for extra_word in extra_words]]
                                potential_names = re.findall(r"\b[A-Z]+\b", " ".join(raw_data))
                                if len(potential_names) == 0:
                                   return " ".join(raw_data)
                                else:
                                        potential_names = [name for name in potential_names if len(name) > 1]
                                        potential_names = set(potential_names)
                                        potential_names = list(potential_names)
                                        return " ".join(potential_names)
                    
                    else:
                            
                                return " ".join(max_size) 

             elif len(max_size) < 60:

                    max_size = [word for word in max_size if word not in [extra_word.lower() for extra_word in extra_words]]
                    max_size = [word for word in max_size if word not in [extra_word for extra_word in extra_words]]
                    spacy_result = detect_names(" ".join(max_size))

                    
                    if len(spacy_result) == 0:
                        text = " ".join(max_size)
                        
                        print("From Capital Regex")
                        potential_names = re.findall(r"\b[A-Z]+\b", text)
                        filtered_names = [name for name in potential_names if len(name) > 1]
                        filtered_names = set(filtered_names)
                        filtered_names = list(filtered_names)
                        if len(filtered_names) == 0:
                            print("From Flow Regex")      
                            flow_names = re.findall(r"^[A-Z][a-z]*(?: [A-Z][a-z]*)?(?: [A-Z][a-z]*)?$", text)
                            
                            if len(flow_names) > 3:
                                print("From Flow Regex") 
                                flow_names = flow_names[:3]
                                return " ".join(flow_names)

                            elif 0 < len(flow_names) < 3:
                                print("From Flow Regex") 
                                return " ".join(flow_names) 
                                            
                            else:
                                print("Nothing")
                                
                                raw_data = raw_data.split()[:3]
                                raw_data = [word for word in raw_data if word not in [extra_word.lower() for extra_word in extra_words]]
                                raw_data = [word for word in raw_data if word not in [extra_word for extra_word in extra_words]] 
                                return " ".join(raw_data)
                            
                        elif len(filtered_names) <= 3:
                            return " ".join(filtered_names)
                        

                        elif len(filtered_names) > 3:
                            filtered_names = filtered_names[:4]
                            return " ".join(filtered_names)
                    else:
                        print("From Spacy")
                        return " ".join(spacy_result) 
            
             elif len(max_size) == 0:
                        print("Nothing")
                        raw_data = raw_data.split()[:4]
                    
                        raw_data = [word for word in raw_data if word not in [extra_word.lower() for extra_word in extra_words]]
                        raw_data = [word for word in raw_data if word not in [extra_word for extra_word in extra_words]]
                        potential_names = re.findall(r"\b[A-Z]+\b", " ".join(raw_data))
                        if len(potential_names) == 0:
                        
                             return " ".join(raw_data)
                        else:
                            potential_names = [name for name in potential_names if len(name) > 1]
                            potential_names = set(potential_names)
                            potential_names = list(potential_names)
                            
                            return " ".join(potential_names)

          except Exception as e:
              print("******detect_names_all*****")
              print(str(e))         





        
        def remove_exact_substring_matches(dictionary):
            print("✅ Inside remove_exact_substring_matches function")
            keys_to_remove = []
            keys = list(dictionary.keys())
            for i in range(len(keys)):
                current_key = keys[i]
                for j in range(len(keys)):
                    if i != j:  
                        other_key = keys[j]
                        if current_key in other_key:
                            keys_to_remove.append(current_key)
                            break 
            for key in keys_to_remove:
                del dictionary[key]
            return dictionary
        

        def remove_exact_unwanted_Keys(data):
            print("✅ Inside remove_exact_unwanted_Keys function")
            to_delete = []

            for key in data:
                words = key.split()
                lengths = [len(word) for word in words]
                
                if all(length <= 3 for length in lengths):
                    to_delete.append(key)
                

            for key in to_delete:
                del data[key]
            return data
        



        def frequent_dynamic_ngrams(text, combined_data):
            try:
                print("✅ Inside frequent_dynamic_ngrams function")
                words = text.split()
                words = [word for word in words if word not in combined_data]
                all_frequent_ngrams = {}
                n = 4  

                while True:
                    ngram_freq = frequent_ngrams(words, n)
                    # print(ngram_freq)
                    if not ngram_freq:
                        break
                    all_frequent_ngrams[f"{n}-gram"] = ngram_freq
                    n += 1

                merged_dict = {}
                for ngram_dict in all_frequent_ngrams.values():
                    merged_dict.update(ngram_dict)

                merged_dict1 = remove_exact_substring_matches(merged_dict)
                remove_2 = remove_exact_unwanted_Keys(merged_dict1)
                return remove_2
            
            except Exception as e:
                print("frequent_dynamic_ngrams", str(e))
                return {}



    
        def get_tables(pdf_file):
            try: 
                print("✅ Inside get_tables function")
                with pdfplumber.open(BytesIO(pdf_file)) as f:
                    for i in f.pages:
                        tables = i.extract_tables()
                        if tables:
                            return "table"
                
                # If no tables are found in any of the pages, return None
                return "no table"
            except Exception as e:
                print("*********tables**************")
                print(str(e))
                return ""

        def get_tables2(pdf_file):
            try: 
                print("✅ Inside get_tables2 function")
                pdf_path = BytesIO(pdf_file)
                # specify the path to your PDF file and the page number containing the table
                
                # pdf_path="resumes/ARUN THOTA RESUME - ARUN THOTA.pdf"
                # iterate over all the pages in the PDF and extract the tables
                for page_number in range(1, len(tabula.read_pdf(pdf_path, pages='all')) + 1):
                    # print("hello")
                    table = tabula.read_pdf(pdf_path, pages=page_number)
                    if table is not None:
                        # print(f"Table on page {page_number}:")
                        # print(table)
                        return ("table");
                    else:
                        return("no table")
                return("no table")
            except Exception as e:
                print("**************tables*************************")
                print(str(e))
                return ""

        def check_Images(pdf_file):
            try: 
                print("✅ Inside check_Images function")
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


    # #Detecting bullets from resume     
        def get_bullets(text):
            
            """
            This function takes a text and returns a list of bullets found in the text.
            It uses regular expressions to detect different types of bullets.
            The function returns the list of bullets, the total number of bullets and a flag indicating whether the bullets are correctly formatted or not.
            The flag is set to 1 if the bullets are correctly formatted and 0 otherwise.
            """
            try:
                print("✅ Inside get_bullets function")
                pattern1 = r"[^\w][\s]\d{1,2}\)+"
                pattern2 =  r"\s?[0-9]\.\s+"
                pattern3 =  r"\s[a-z]\)\s+"
                pattern4 =  r"[^\w]•\s+|●\s+|✓\s+|▪\s+|❖\s+|➢\s+|[^\w]\*\s+|"
            
                pattern5 =  r"\s[A-Z]\)\s+"
                pattern6 =   r"[^\w][IVX]+\.\s+" 

                finalBulletSet = set()
                finalBullet = set()

                bullets1 = re.findall(pattern1,text)
                if len(bullets1) > 0:
                    finalBulletSet.add(bullets1[0])
                


                bullets2 = re.findall(pattern2,text)
                if len(bullets2) > 0:
                    finalBulletSet.add(bullets2[0].strip())
                bullets3 = re.findall(pattern3,text)
                if len(bullets3) > 0:
                    finalBulletSet.add(bullets3[0])

                bullets4 = re.findall(pattern4,text)

                bulletSet = set(bullets4)
                finalBulletSet = finalBulletSet | bulletSet
                
                

                bullets5 = re.findall(pattern5,text)
                if len(bullets5) > 0:
                    finalBulletSet.add(bullets5[0])

                bullets6 = re.findall(pattern6,text)
                if len(bullets6) > 0:
                    finalBulletSet.add(bullets6[0])

                for x in finalBulletSet:
                    s = x.replace('\n','')
                    z = s.replace('\x0c','')
                    finalBullet.add(z.replace(' ',''))
                    
                newList = list(finalBullet) 
                updatedList = [x for x in newList if x != ""]
                finalBullet1 = set(updatedList)
                Bullets_Total = len(finalBullet1) 
                flag_Bullet=1
                if Bullets_Total > 4 or Bullets_Total==0:
                    flag_Bullet = 0
                

                else:
                    for i in finalBullet1:
                        if i == '●' or i == '•' or i == '▪':
                            continue
                        else:
                            flag_Bullet = 0
                            break


            
                return list(finalBullet1),Bullets_Total,flag_Bullet
            except Exception as e :
                print("get_bullets ",e)
                return [],0,0

    
    #     #Extracting ATS friendly dates     
        def getATS_dates(text):
            """
            This function takes a text and returns a list of ATS friendly dates found in the text.
            It uses regular expressions to detect different types of dates.
            The function returns the list of dates and the total number of dates found.
            """
            try:
                print("✅ Inside getATS_dates function")
                z = []
                ats = []
                r = []
                x = []
                n = []
                all6 = re.findall(r"([\d]{4}\s?[-|—|–]\s?[\d]{4}[^0-9])",text)
                for i in all6:
                    n1 = i.replace('(','')
                    n2 = n1.replace(')','')
                    n2 = n2.strip()
                    n.append(n2)


                all8  = re.findall(r"\b\d{1,2}[/]\d{4}\b\s?[-|–|—]\s?\b\d{1,2}[/]\d{4}\b", text)

                all2 = re.findall(r"\b(\d{2}/\s?\d{4}) [-|–|—] (Present|Current|till date)\b", text, re.IGNORECASE)
                
                for i in all2:
                    if i[1] not in i[0]: 
                        u = i[0]+' - '+i[1]
                        r.append(u)
                    else:
                        for j in i:
                            r.append(j) 
                            break  

                z = re.findall(r'(?<![\d-])(?:(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s?\d{4})(?!\s*[-–—]\s*(?:PRESENT|present|CURRENT|current|TILL DATE|till date))(?:\s*[-–—]\s*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s?\d{4})?(?![\d-])', text,re.IGNORECASE)

            
                all = re.findall(r"\b([\w]{3} \d{4} [-–—] (PRESENT|CURRENT|TILL DATE))\b", text, re.IGNORECASE)
                
                if len(all) > 0:
                    for i in all:
                        if i[1] not in i[0]:
                            a = i[0]+' - '+i[1]
                            x.append(a)    
                        else:
                            for j in i:
                                x.append(j) 
                                break  

            
                
                all10 =  re.findall(r"(\d{2}[/]\d{2}[/]\d{2,4}\s[–|-|—]\s\d{2}[/]\d{2}[/]\d{2,4})",text)
                ats = n+all8+z+all10+r+x
                new_ats=[]
                for i in ats:
                    i=i.replace("\n"," ")
                    i=i.strip()
                    new_ats.append(i)
                return list(new_ats)
            
            except Exception as e:
                print("getATS_dates ",e)
                return []

    #     #Extracting Non-ATS  dates 
        def get_nonATSdates(text):
            
            """
            Get all the non-ATS dates from a given text.

            This function will use regular expressions to find all the dates that do not follow the ATS date format.
            It will return a list of all the non-ATS dates found in the text.

            Parameters
            ----------
            text (str): The text to search for non-ATS dates.

            Returns
            -------
            list: A list of all the non-ATS dates found in the text.

            """
            try: 
                print("✅ Inside get_nonATSdates function")
                ats = getATS_dates(text)
                t = []
                for i in ats:
                    if '-' in i:
                        x = i.split("-")
                        for j in x:
                            t.append(j.strip())  
                    elif '/' in i:
                        x = i.split("-")
                        for j in x:
                            t.append(j.strip()) 
                    elif '/' in i:
                        x = i.split("–")
                        for j in x:
                            t.append(j.strip())  
                

                # 01–01–2002
                all10 =  re.findall(r"(\d{2}[/]\d{2}[/]\d{2,4}\s[–-—]\s\d{2}[/]\d{2}[/]\d{2,4})",text)
                if len(all10) == 0:
                    al = re.findall(r"\b\d{1,2}\s?[/]\s?\d{1,2}\s?[/]\s?\d{4}\b", text)

                all = re.findall(r"\b\d{1,2}\s?[–-]\s?\d{1,2}\s?[–-]\s?\d{4}\b", text)

                # all_ = re.findall(r"\d{1,2}\s[/–]\s\d{1,2}\s[/–]\s\d{4}",text)

                # 2004–11–11
                all1 =  re.findall(r"\b(?!.*\d{1,2}\s?[-–/]\s?\d{1,2}\s?[-–/]\s?\d{4})\d{4}\s?[-–/]\s?\d{1,2}\s?[-–/]\s?\d{1,2}\b",text)
                # all1_ = re.findall(r"\d{4}\s?[/–-]\s?\d{1,2}\s?[/–-]\s?\d{1,2}",text)

                # 20 11 2011
                all2 = re.findall(r"\d{1,2}\s?(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}", text, re.IGNORECASE)
            
                
            
                # May 08, 2004
                all3 = re.findall(r"\b((JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s[\d]{1,2}[,]\s[\d]{4})\b",text)
                all3_ = [] 
                if len(all3) > 0:
                    for i in all3:
                        all3_.append(i[0])

            
                
                
                # 08 June, 2004
                all4 = re.findall(r"([\d]{1,2}\s(JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[,]\s[\d]{4})",text)
                all4_ = [] 
                if len(all4) > 0:
                    for i in all4:
                        all4_.append(i[0])
                
                # 2022, January 12
                all5 = re.findall(r"([\d]{4}[,]\s(JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s[\d]{1,2})",text)  
                all5_ = [] 
                if len(all5) > 0:
                    for i in all5:
                        all5_.append(i[0])
                
            
                all7 = re.findall(r"((?<!\S)(?!.-.\b)(?:JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[,]\s[\d]{4})(?!\S)", text)



                all7_ = [] 
                if len(all7) > 0:
                    for i in all7:
                        all7_.append(i[0])

                # 14 2004
                new1 =  re.findall(r"(\b\d{1,2}\s\d{4})\b", text)
            
                # 01/2002 or 01-2002 or 01.2020
                new2 = re.findall(r"(?<!\d\d/)(?<!\d)(?!\d{2}/\d{2}/\d{4})([\d]{1,2}[.–/][\d]{4})(?=[^0-9]|$)(?<!\d\d)",text)
                
            

                new21 = [ele for ele in new2 if ele not in t]
            
                # 2022/05
                new3 = re.findall(r"([\d]{4}[/][\d]{1,2})(?=[^0-9\s]|$)",text)

                #  2022/05 to 2022/06
                new4 = re.findall(r"(\d{2,4}[/]\d{1,2}\s[t][o]\s\d{2,4}[/]\d{1,2})(?=[^0-9\s]|$)",text)


                new5 = re.findall(r"\d{1,2}[\s?][-](JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s?][-][\d]{4}",text)
                new5_ = [] 
                if len(new5) == 0:
                    new51_= re.findall(r"((JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s?[-]\s?[\d]{4})", text,re.IGNORECASE)
                
                if len(new51_) > 0:
                    for i in new51_:
                        new5_.append(i[0])
                    
                
                
                new7 = re.findall(r"\b(\d{1,2}[\s]?[/–-][\s]?(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s]?[/–-][\s]?\d{4})\b", text,re.IGNORECASE)
                new7_ = [] 
                if len(new7) > 0:
                    for i in new7:
                        new7_.append(i[0])
            
            
                new6 = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", text) 
                res = al + all + all1  + all2 + all3_ + all4_ + all5_ + all7_ + new1 + new21 + new3 + new4+ new7_+ new5 + new5_
                res1 = []
                if len(res) == 0 and len(ats) == 0:
                    res1 = res + new6
                else:
                    res1 = res   

                        
                return list(res1)
            except Exception as e:
                print("get_nonATSdates ",e)
                return []

    #     #Extracting Measurable Data without preprocessing using named Entity
        def get_namedEntityMeasurable(text):
            """
            Extracts measurable entities (such as numbers, quantities, percentages, money, and other cardinal values) 
            from a given text using regular expressions and named entity recognition (NER) via SpaCy.

            This function performs the following tasks:
            1. Identifies large numbers followed by 'million', 'billion', or 'trillion' (e.g., '5 billion').
            2. Preprocesses the input text by cleaning unwanted characters and symbols.
            3. Uses SpaCy's NER to identify and extract relevant measurable entities such as:
            - Percentages
            - Money values
            - Cardinal numbers
            - Ordinal numbers
            - Quantities

            Args:
                text (str): The input text from which measurable entities will be extracted.

            Returns:
                list: A list of measurable entities (strings) found in the input text.
                    These can include numbers, percentages, money, cardinal and ordinal values, and quantities.
                
            Example:
                text = "The total sales were 5 million dollars and the discount rate was 20%."
                measurable_entities = get_namedEntityMeasurable(text)
                print(measurable_entities)  # Output: ['5 million', '20%']
            """
            try:
                print("✅ Inside get_namedEntityMeasurable function")
                # print("here at mesaurableee")
                mesurable=[]
                new_textx = re.findall(r"([\d]{1,4}(million|billion|trillion))",text)
                if len(new_textx) > 0:
                    for i in new_textx:
                        mesurable.add(i[0])


                
                #preprocessing data
                text = text.replace('\n'," ")
                text = text.replace('('," ")
                pattern = r"\s[0-9]\)+\s?"
                new_text= re.sub(pattern, "", text)
                pattern1 = r"[0-9][.][a-zA-Z]"
                pattern2 = r"\b[0-9]{1,4}[a-zA-Z]\b"
                pattern3 = r"[a-zA-Z][.][0-9]"
                new_text0 = re.sub(pattern1, "",  new_text)
            
                new_text1= re.sub(pattern2, "",  new_text0)
                new_text2= re.sub(pattern3, "",  new_text1)

                new_text2 = new_text2.replace(')'," ")
                # Process the text with Spacy

                doc = nlp(new_text2)




                # Iterate through named entities
                for entity in doc.ents:
                

                    if (entity.label_ == 'PERCENT' or entity.label_ == 'MONEY' or entity.label_ == 'CARDINAL' or entity.label_ == 'ORDINAL' or entity.label_ == 'QUANTITY'):
                    
                        mesurable.append(entity.text)
                
                return mesurable
            except Exception as e:
                print("get_namedEntityMeasurable ",e)
                measurable=[]
                return measurable


    #     # Pre Processing measurable data coming from named Entities"    OLD


        # def get_measurableUpdated(text,finalBullet,measurable,phone_all1,dates):
        #     try:
            
        #         finalBullet_list  = list(finalBullet)
        #         phone_all1_list = list(phone_all1)
        #         measurable_updated = [x for x in measurable if x not in finalBullet_list]
        #         measurable_updated = [x for x in measurable_updated if x not in phone_all1_list]
        #         measurable_updated = [x for x in measurable_updated if x not in dates]
        #         measurable_updated = [x for x in measurable_updated if "#" not in x]
        #         measurable_updated = [x for x in measurable_updated if '.com' not in x]
        #         measurable_updated = [x for x in measurable_updated if '.in' not in x]
        #         measurable_updated = [x for x in measurable_updated if '.io' not in x]
                
        #         pattern2 =  r"\s[0-9]{1,2}\.\s+"
        #         bullets2 = re.findall(pattern2,text)
        #         bullet2_new = []
        #         for x in bullets2:
        #             s = x.replace('\n','')
        #             z = s.replace('\x0c','')
        #             bullet2_new.append(z.replace(' ',''))
            
        #         measurable_updated = [x for x in measurable_updated if x not in bullet2_new]

        #         list1 = ['1','2','3','4','5','6']
        #         measurable_updated = [x for x in measurable_updated if x not in list1]
        #         measurable_updated = [x for x in measurable_updated if '\uf0fc' not in x]
        #         class1 = ['10th','12th','10TH','12TH']
        #         measurable_updated_new = []  
            
        #         for i in measurable_updated:
                
        #             if i.isdigit() and len(i) >= 6:
                
        #                 continue
        #             elif '/' in i:
                
        #                 pattern = r"([\d]{1,2}\s?[/]\s?[\d]{4})"
        #                 x = re.findall(pattern,i)
                    
        #                 if len(x)>0:
                    
        #                     continue
        #                 else:
        #                     measurable_updated_new.append(i)
        #                     continue

        #             elif i in class1:
                
        #                 continue

        #             elif len(i) >=  15:
                    
        #                 continue
                
        #             else:
                
        #                 measurable_updated_new.append(i)
                

            
            
        #         measurable_updated_change = []
        #         file_path = 'Measurable.xlsx'
        #         dataframe1 = pd.read_excel(file_path,usecols='A')
        #         # dataframe1 = pd.read_excel('Named Entity.xlsx',usecols='A')
        #         SKILLS_DB = dataframe1.values.tolist()
        #         topics_flat1 = [topic for sublist in SKILLS_DB for topic in sublist]
        #         topics_flat =  [x.lower() for x in topics_flat1]

        #         flag = 1
            
        #         for i in measurable_updated_new:
        #             for j in i:
        #                 if j.lower() >= 'a' and j.lower() <= 'z':
        #                     continue
        #                 else:
                
        #                     flag = 0
        #                     break
        #             if flag == 0:
                
        #                 flag = 1
        #                 measurable_updated_change.append(i)
        #             else:
                
        #                 if i.lower() in topics_flat:
                    
        #                     measurable_updated_change.append(i.lower())

        #         measurable_updated_change_new = []        
        #         new_ = re.findall(r"([\d]{1,2}\s(JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))",text)
        #         new = [] 
        #         if len(new_) > 0:
        #             for i in new_:
        #                 new.append(i[0])

        #         measurable_updated_change_new = [x for x in measurable_updated_change if x not in new]  
        #         measurable_updated_change_new1 = [elem for elem in measurable_updated_change_new if not re.search(r"[\d]\s?[-–]", elem)]
        #         new1_ = re.findall(r"\d+\+",text)
        #         if len(new1_) > 0:
        #             measurable_updated_change_new1.extend(new1_)
        #         final_ans=measurable_updated_change_new1
        #         return final_ans
        #     except Exception as e:
        #         print("Measurable Updated Cleaning",e)
        #         return []

          



        def get_measurableUpdated(text,finalBullet,measurable,phone_all1,dates):
            """
            This function processes and filters a list of measurable items based on various criteria, 
            returning a cleaned and updated list of measurable items that are relevant to the provided 
            text data. It applies several filters to remove irrelevant items, such as specific keywords, 
            dates, and predefined lists. The function also checks for the presence of certain words in 
            a reference database, as well as removing duplicates and unwanted patterns from the final output.

            Parameters:
            - text (str): The input text that will be analyzed for specific patterns and keywords.
            - finalBullet (iterable): A list or iterable of items to exclude from the measurable list.
            - measurable (list): The list of measurable items to be filtered and updated.
            - phone_all1 (iterable): A list or iterable of phone-related or irrelevant items to be excluded.
            - dates (iterable): A list or iterable of date-related strings or patterns to exclude.
            
            Returns:
            - list: A list of cleaned and updated measurable items that pass all filters.
            
            Process:
            1. The function removes items from the measurable list that are present in `finalBullet`, 
            `phone_all1`, or `dates`.
            2. It filters out strings that contain specific keywords like "#", ".com", ".in", and ".io".
            3. It removes numeric items, items with certain patterns (such as date formats), and unwanted 
            class-related keywords (e.g., "10th", "12th").
            4. The function then checks for items containing more than a certain length, excluding those.
            5. After these general filters, the function looks for matches with a reference database 
            (`Measurable.xlsx`) to identify valid skills or topics.
            6. Additional cleaning is applied by removing known date patterns and other irrelevant patterns 
            (e.g., numerical ranges with "-").
            7. Duplicates are removed, and the final list is returned.

            Exception Handling:
            - If any exception occurs during the execution of the function, it logs the error message 
            and returns an empty list.

            Example:
            >>> get_measurableUpdated("Some text here", ["item1", "item2"], ["item3", "item4"], ["phone"], ["2020-01-01"])
            ['item3', 'item4']
            """
            try:
                
                print("✅ Inside get_measurableUpdated function")
                finalBullet_list  = list(finalBullet)
                phone_all1_list = list(phone_all1)
                measurable_updated = [x for x in measurable if x not in finalBullet_list]
                measurable_updated = [x for x in measurable_updated if x not in phone_all1_list]
                measurable_updated = [x for x in measurable_updated if x not in dates]
                measurable_updated = [x for x in measurable_updated if "#" not in x]
                measurable_updated = [x for x in measurable_updated if '.com' not in x]
                measurable_updated = [x for x in measurable_updated if '.in' not in x]
                measurable_updated = [x for x in measurable_updated if '.io' not in x]
                
                pattern2 =  r"\s[0-9]{1,2}\.\s+"
                bullets2 = re.findall(pattern2,text)
                bullet2_new = []
                for x in bullets2:
                    s = x.replace('\n','')
                    z = s.replace('\x0c','')
                    bullet2_new.append(z.replace(' ',''))
            
                measurable_updated = [x for x in measurable_updated if x not in bullet2_new]

                list1 = ['1','2','3','4','5','6']
                measurable_updated = [x for x in measurable_updated if x not in list1]
                measurable_updated = [x for x in measurable_updated if '\uf0fc' not in x]
                class1 = ['10th','12th','10TH','12TH']
                measurable_updated_new = []  
            
                for i in measurable_updated:
                
                    if i.isdigit() and len(i) >= 6:
                
                        continue
                    elif '/' in i:
                
                        pattern = r"([\d]{1,2}\s?[/]\s?[\d]{4})"
                        x = re.findall(pattern,i)
                    
                        if len(x)>0:
                    
                            continue
                        else:
                            measurable_updated_new.append(i)
                            continue

                    elif i in class1:
                
                        continue

                    elif len(i) >=  15:
                    
                        continue
                
                    else:
                
                        measurable_updated_new.append(i)
                

            
            
                measurable_updated_change = []
                file_path = 'Measurable.xlsx'
                dataframe1 = pd.read_excel(file_path,usecols='A')
                # dataframe1 = pd.read_excel('Named Entity.xlsx',usecols='A')
                SKILLS_DB = dataframe1.values.tolist()
                topics_flat1 = [topic for sublist in SKILLS_DB for topic in sublist]
                topics_flat =  [x.lower() for x in topics_flat1]

                flag = 1
            
                for i in measurable_updated_new:
                    for j in i:
                        if j.lower() >= 'a' and j.lower() <= 'z':
                            continue
                        else:
                
                            flag = 0
                            break
                    if flag == 0:
                
                        flag = 1
                        measurable_updated_change.append(i)
                    else:
                
                        if i.lower() in topics_flat:
                    
                            measurable_updated_change.append(i.lower())

                measurable_updated_change_new = []        
                new_ = re.findall(r"([\d]{1,2}\s(JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))",text)
                new = [] 
                if len(new_) > 0:
                    for i in new_:
                        new.append(i[0])

                measurable_updated_change_new = [x for x in measurable_updated_change if x not in new]  
                measurable_updated_change_new1 = [elem for elem in measurable_updated_change_new if not re.search(r"[\d]\s?[-–]", elem)]
                new1_ = re.findall(r"\d+\+",text)
                measurable_string = []
                if len(new1_) > 0:
                    measurable_updated_change_new1.extend(new1_)
                final_ans=  set(measurable_updated_change_new1)
                
                
                measure_dict = {}
                if len(final_ans) > 0:
                  for i in final_ans:
                    found_matches = find_words(text.split(),i.split())
                    measure_dict[i] = found_matches
                    if found_matches:
                        measurable_string.append(found_matches)
                
                measurable_output = remove_duplicates(measurable_string)
                final_measurable = []
                for value in measurable_output:
                    for key, val in measure_dict.items():
                        if val == value:
                            final_measurable.append(key)
                
                # print(measure_dict)
                # print(final_measurable)
                return list(final_measurable)
            except Exception as e:
                print("Measurable Updated Cleaning",e)
                return []

        def remove_duplicates(input_list):
            print("✅ Inside remove_duplicates function")
            result = []
            for i, s1 in enumerate(input_list):
                found_duplicate = False
                for j, s2 in enumerate(input_list[i+1:], start=i+1):
                    if s1 in s2 or s2 in s1:  
                        found_duplicate = True
                        break
                if not found_duplicate:
                    result.append(s1)
            return result



        def find_words(input_list, search_list):
            
            print("✅ Inside find_words function")
    
            for i in range(len(input_list)):
            
                list_input = [word.rstrip('.') for word in input_list[i:i+len(search_list)]]
                list_input = [word.lstrip('(') for word in list_input]
                list_input = [word.rstrip(')') for word in list_input]
                list_input = [word.rstrip(',') for word in list_input]

                if list_input == search_list:
            
                    max_context = min(i, 3) 
                    min_context = min(len(input_list) - i - len(search_list), 3) 
                    output_string = ""
                    

                    for j in range(max_context):
                        output_string += input_list[i - max_context + j] + " "

                    output_string += " ".join(search_list) + " "
                    for j in range(min_context):
                        output_string += input_list[i + len(search_list) + j] + " "

                    return output_string.strip()

            return ""


        

    #     # Extracting  phone numbers in resume using text
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



        #Extracting file details i.e. size,name,type
        def get_fileDetails(pdf_file):
            try:
                print("✅ Inside get_fileDetails function")
                file_size = pdf_file['ContentLength']
                file_size_kb = round(file_size/1024)
                
                flag_file_size = 1 if file_size_kb > 2*1024 else 0

                file_type = pdf_file['ContentType']

                
                return file_size_kb,file_type,flag_file_size
            
            except Exception as e:
                print("***********file details*************")
                print(str(e))
                return 0,'',0
            

            
        def score(standard_font_style_flag,multiple_font_style,multiple_font_size,content_size_flag,font_colors_Total,standard_color_flag,actionwords_total,actionwords_total_negative,total_repeated_actionwords_negative,Bullets_Total,standard_bullet_flag,total_repeated_actionwords,ats_date,dates_nonAts,clean_measurable,total_word_count,email_finderSet,phonenumbers_finderSet,images,linkedIn_flag,pages_count,personalPronouns,tables_flag,Work_Project_Headings,EduSkill_Headings,NRlength,ORlength_db,Skills_Total,standard_match_headings_count,sectionMapCount,phone_all1,actualHeadingsCount,experience,fillerwords_total,output_voice,repeated_words,file_name, name_output):
                try:
                
                    print("✅ Inside score function")
                    score_array = []
                    scoring  = 0
                    scoring1 = 0
                    if standard_font_style_flag == 1:
                        scoring += 3
                        scoring1 = 3
                    else:
                        scoring += 0
                        scoring1 = 0

                    score_array.append(scoring1)  
                    print(scoring)


                    scoring1 = 0
                    if multiple_font_style == 1 or multiple_font_style == 2:
                        scoring += 2
                        scoring1 = 2
                    elif multiple_font_style == 3 or multiple_font_style == 4:    
                        scoring += 1
                        scoring1 = 1

                    else:
                        scoring += 0
                        scoring1 = 0   
                    

                    score_array.append(scoring1)
                    print(scoring)


                    scoring1 = 0
                    if multiple_font_size == 1 or multiple_font_size>6:
                        scoring -= 2
                        scoring1 = -2
                    if multiple_font_size == 3  or multiple_font_size == 4:    
                        scoring += 5
                        scoring1 = 5
                    if multiple_font_size == 5  or multiple_font_size == 6 or multiple_font_size == 2:    
                        scoring += 2
                        scoring1 = 2

                    print(scoring)
                    score_array.append(scoring1)
                    
                    scoring1 = 0
                    if content_size_flag == 1:
                        scoring += 6
                        scoring1 = 6
                    else:
                        scoring -= 3 
                        scoring1 = -3
                    print(scoring)
                    score_array.append(scoring1)
                    
                    scoring1 = 0
                    if font_colors_Total < 5:
                        scoring += 1
                        scoring1 = 1
                    elif font_colors_Total == 5:
                        scoring += 1
                        scoring1 = 1
                    else:
                        scoring += 0    
                        scoring1 = 0
                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if standard_color_flag == 1:
                        scoring += 1
                        scoring1 = 1
                    else:
                        scoring += 0    
                        scoring1 = 0
                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if experience == 0:
                        if len(Work_Project_Headings) == 2:
                            scoring += 4
                            scoring1 = 4
                        elif len(Work_Project_Headings) == 1:
                            scoring += 4
                            scoring1 = 4
                        else:
                            scoring -= 5
                            scoring1 = -5

                    elif experience <2:
                        if len(Work_Project_Headings) == 2:
                            scoring += 4
                            scoring1 = 4
                        elif len(Work_Project_Headings) == 1:
                            scoring += 2
                            scoring1 = 2
                        else:
                            scoring -= 5
                            scoring1 = -5

                    elif experience >= 2 and experience<=5:
                        if len(Work_Project_Headings) == 2:
                            scoring += 4
                            scoring1 = 4
                        elif len(Work_Project_Headings) == 1:
                            scoring += 2
                            scoring1 = 2
                        else:
                            scoring -= 5
                            scoring1 = -5

                    elif experience >5:
                        if len(Work_Project_Headings) == 2:
                            scoring += 4
                            scoring1 = 4
                        elif len(Work_Project_Headings) == 1:
                            scoring += 2
                            scoring1 = 2
                        else:
                            scoring -= 5
                            scoring1 = -5
                    else:
                        scoring+=0
                        scoring1 = 0


                    print(scoring)
                    score_array.append(scoring1)
                    
                    scoring1 = 0
                    if len(EduSkill_Headings) == 2:
                        scoring += 5
                        scoring1 = 5
                    elif len(EduSkill_Headings) == 1:
                        scoring += 2
                        scoring1 = 2
                    else:
                        scoring -= 5
                        scoring1 = -5

                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if experience==0:
                        if actionwords_total > 10:
                            scoring += 9
                            scoring1 = 9

                        elif actionwords_total >= 4 and actionwords_total <= 10: 
                            value = ((actionwords_total)/11)*9
                            scoring += value
                            scoring1 = value
                        else:
                            scoring -= 2
                            scoring1 = -2        
                    elif experience<2:
                        if actionwords_total > 10:
                            scoring += 9
                            scoring1 = 9
                        elif actionwords_total >= 4 and actionwords_total <= 10: 
                            value = ((actionwords_total)/11)*9
                            scoring += value
                            scoring1 = value
                        else:
                            scoring -= 2
                            scoring1 = -2
                    elif experience>=2 and experience<=5:
                        if actionwords_total > 10:
                            scoring += 9
                            scoring1 = 9

                        elif actionwords_total >= 4 and actionwords_total <= 10: 
                            value = ((actionwords_total)/11)*9
                            scoring += value
                            scoring1 = value

                        else:
                            scoring -= 2
                            scoring1 = -2
                    elif experience>5:
                        if actionwords_total > 10:
                            scoring += 9
                            scoring1 = 9

                        elif actionwords_total >= 4 and actionwords_total <= 10: 
                            value = ((actionwords_total)/11)*9
                            scoring += value
                            scoring1 = value
                        else:
                            scoring -= 2
                            scoring1 = -2
                    else:
                        scoring+=0
                        scoring1 = 0
                    
                    print(scoring)
                    score_array.append(scoring1)
                    
                    scoring1 = 0
                    if NRlength > 2:
                        scoring -= 6
                        scoring1 = -6
                    elif NRlength == 2:      
                        scoring -= 4
                        scoring1 = -4

                    elif NRlength == 1:    
                        scoring -= 2
                        scoring1 = -2
                    else:
                        scoring += 0    
                        scoring1 = 0
                    print(scoring)
                    
                    score_array.append(scoring1)

            #changed
                    scoring1 = 0
                    if experience==0:
                        if ORlength > 2:
                            scoring += 5
                            scoring1 = 5
                        elif ORlength == 1 or ORlength == 2:
                            scoring += 2
                            scoring1 = 2
                        else:
                            scoring += 0
                            scoring1 = 0

                    elif experience<2:
                        if ORlength > 2:
                            scoring += 5
                            scoring1 = 5
                        elif ORlength == 1 or ORlength == 2:
                            scoring += 3
                            scoring1 = 3
                        else:
                            scoring += 0
                            scoring1 = 0

                    elif experience>=2 and experience<=5:
                        if ORlength > 1:
                            scoring += 5
                            scoring1 = 5
                        elif ORlength == 1 :
                            scoring += 2
                            scoring1 = 2
                        else:
                            scoring += 0
                            scoring1 = 0
                    
                    elif experience>5:
                        if ORlength > 1:
                            scoring += 5
                            scoring1 = 5
                        elif ORlength == 1 :
                            scoring += 2
                            scoring1 = 2
                        else:
                            scoring += 0
                            scoring1 = 0
                    else:
                        scoring+=0
                        scoring1 = 0


                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if experience == 0:
                        if total_repeated_actionwords > 2:
                            scoring -= 3
                            scoring1 = -3
                        elif total_repeated_actionwords == 2:            
                            scoring -= 2
                            scoring1 = -2
                        elif total_repeated_actionwords == 1:            
                            scoring -= 2
                            scoring1 = -2
                        else:
                            scoring += 0
                            scoring1 = 0

                    elif experience <2:
                        if total_repeated_actionwords > 2:
                            scoring -= 3
                            scoring1 = -3
                        elif total_repeated_actionwords == 2:            
                            scoring -= 2
                            scoring1 = -2
                        elif total_repeated_actionwords == 1:            
                            scoring -= 2
                            scoring1 = -2
                        else:
                            scoring += 0
                            scoring1 = 0
                    
                    elif experience >=2 and experience<=5:
                        if total_repeated_actionwords > 2:
                            scoring -= 3
                            scoring1 = -3
                        elif total_repeated_actionwords == 2:            
                            scoring -= 2
                            scoring1 = -2
                        elif total_repeated_actionwords == 1:            
                            scoring -= 2
                            scoring1 = -2
                        else:
                            scoring += 0
                            scoring1 = 0
                    elif experience >5:
                        if total_repeated_actionwords > 2:
                            scoring -= 3
                            scoring1 = -3
                        elif total_repeated_actionwords == 2:            
                            scoring -= 2
                            scoring1 = -2
                        elif total_repeated_actionwords == 1:            
                            scoring -= 2
                            scoring1 = -2
                        else:
                            scoring += 0
                            scoring1 = 0
                    else :
                        scoring+=0
                        scoring1 = 0
                    print(scoring)
                    score_array.append(scoring1)
                    
                    scoring1 = 0
                    if Bullets_Total == 0 or Bullets_Total > 2:
                        scoring -= 4
                        scoring1 = -4
                    elif Bullets_Total == 2:
                        scoring -= 3
                        scoring1 = -3
                    else:
                        scoring += 0 
                        scoring1 = 0   


                    print(scoring)
                    score_array.append(scoring1)
                    
                    scoring1 = 0
                    if standard_bullet_flag == 1:
                        scoring += 3
                        scoring1 = 3
                    else:
                        scoring += 0
                        scoring1 = 0

                    print(scoring)     
                    score_array.append(scoring1)

                    ats_length = 1 if len(ats_date) > 0 else 0
                    nonats_length = 1 if len(dates_nonAts) > 0 else 0

                    scoring1 = 0
                    if ats_length == 1:
                        scoring += 5
                        scoring1 = 5
                    else:
                        scoring += 0 
                        scoring1 = 0     

                    print(scoring)
                    score_array.append(scoring1)
                    
                    scoring1 = 0
                    if nonats_length == 1:
                        scoring -= 3
                        scoring1 = -3
                    else:
                        scoring += 0       
                        scoring1 = 0
                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if actionwords_total_negative > 2:
                        scoring -= 4
                        scoring1 = -4
                    elif actionwords_total_negative == 1 or actionwords_total_negative == 2:
                        scoring -= 2
                        scoring1 = -2
                    else:
                        scoring += 0
                        scoring1 = 0

                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if total_repeated_actionwords_negative > 2:
                        scoring -= 2
                        scoring1 = -2
                    elif total_repeated_actionwords_negative == 1 or total_repeated_actionwords_negative == 2:
                        scoring -= 1
                        scoring1 = -1

                    else:
                        scoring += 0
                        scoring1 = 0   

                    print(scoring)   
                    score_array.append(scoring1)

                    scoring1 = 0
                    if linkedIn_flag == 1:
                        scoring += 4
                        scoring1 = 4
                    else:
                        scoring -= 4    
                        scoring1 = -4


                    print(scoring)
                    score_array.append(scoring1)
            #changed
                    scoring1 = 0
                    if experience==0:

                        if len(clean_measurable) >=12:
                            scoring += 9
                            scoring1 = 9

                        elif len(clean_measurable) >=5 and len(clean_measurable) <= 11:
                            new_score = ((len(clean_measurable))/12)*9   
                            scoring += new_score
                            scoring1 = new_score

                        elif len(clean_measurable)<=4:
                            scoring -= 3 
                            scoring1 = -3   
                        else:
                            scoring+=0
                            scoring1 = 0
                    elif experience<2:

                        if len(clean_measurable) >=12:
                            scoring += 9
                            scoring1 = 9

                        elif len(clean_measurable) >=5 and len(clean_measurable) <= 11:
                            new_score = ((len(clean_measurable))/12)*9   
                            scoring += new_score
                            scoring1 = new_score
                        elif len(clean_measurable)<=4:
                            scoring -= 3    
                            scoring1 = -3
                        else:
                            scoring+=0
                            scoring1 = 0
                    elif experience>=2 and experience<=5:

                        if len(clean_measurable) >=12:
                            scoring += 9
                            scoring1 = 9

                        elif len(clean_measurable) >=5 and len(clean_measurable) <= 11:
                            new_score = ((len(clean_measurable))/12)*9   
                            scoring += new_score
                            scoring1 = new_score

                        elif len(clean_measurable)<=4:
                            scoring -= 3    
                            scoring1 = -3
                        else:
                            scoring+=0
                            scoring1 = 0
                    elif experience>5:

                        if len(clean_measurable) >=12:
                            scoring += 9
                            scoring1 = 9

                        elif len(clean_measurable) >=5 and len(clean_measurable) <= 11:
                            new_score = ((len(clean_measurable))/12)*9   
                            scoring += new_score
                            scoring1 = new_score
                        elif len(clean_measurable)<=4:
                            scoring -= 3    
                            scoring1 = -3
                        else:
                            scoring+=0
                            scoring1 = 0
                    else:
                        scoring+=0
                        scoring1 = 0




                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if experience==0:

                        if total_word_count <= 150:
                            scoring -= 8
                            scoring1 = -8
                        elif total_word_count > 150 and total_word_count <= 165: 
                            scoring -= 6
                            scoring1 = -6
                        elif total_word_count > 165 and total_word_count <= 200: 
                            scoring += 2
                            scoring1 = 2
                        elif total_word_count > 200 and total_word_count <= 300: 
                            scoring += 4
                            scoring1 = 4
                        elif total_word_count > 300 and total_word_count <= 450: 
                            scoring += 8
                            scoring1 = 8
                        elif total_word_count > 450 and total_word_count <= 600: 
                            scoring += 4
                            scoring1 = 4
                        else:
                            scoring -= 8
                            scoring1 = -8

                    elif experience<2:

                        
                        if total_word_count <= 165: 
                            scoring -= 8
                            scoring1 = -8
                        elif total_word_count > 165 and total_word_count <= 250: 
                            scoring -= 4
                            scoring1 = -4
                        elif total_word_count > 250 and total_word_count <= 350: 
                            scoring += 6
                            scoring1 = 6
                        elif total_word_count > 350 and total_word_count <= 500: 
                            scoring += 8
                            scoring1 = 8
                        elif total_word_count > 500 and total_word_count <= 700: 
                            scoring += 4
                            scoring1 = 4
                        else:
                            scoring -= 8
                            scoring1 = -8

                    elif experience>=2 and experience<=5:

                        if total_word_count <= 165: 
                            scoring -= 8
                            scoring1 = -8
                        elif total_word_count > 165 and total_word_count <= 250: 
                            scoring -= 4
                            scoring1 = -4
                        elif total_word_count > 250 and total_word_count <= 400: 
                            scoring += 4
                            scoring1 = 4
                        elif total_word_count > 400 and total_word_count <= 750: 
                            scoring += 8
                            scoring1 = 8
                        elif total_word_count > 750 and total_word_count <= 900: 
                            scoring += 4
                            scoring1 = 4
                        else:
                            scoring -= 8
                            scoring1 = -8


                    elif experience>5:

                        if total_word_count <= 165:
                            scoring -= 8
                            scoring1 = -8
                        elif total_word_count > 165 and total_word_count <= 200: 
                            scoring -= 4
                            scoring1 = -4
                        elif total_word_count > 200 and total_word_count <= 300: 
                            scoring -= 2
                            scoring1 = -2
                        elif total_word_count > 300 and total_word_count <= 450: 
                            scoring += 2
                            scoring1 = 2
                        elif total_word_count > 450 and total_word_count <= 950: 
                            scoring += 8
                            scoring1 = 8
                        elif total_word_count > 950 and total_word_count <= 1000: 
                            scoring += 6
                            scoring1 = 6
                        
                        else:
                            scoring -= 8
                            scoring1 = -8
                    
                    else :
                        scoring+=0
                        scoring1 = 0
                    print(scoring)
                    score_array.append(scoring1)

            #changed
                    scoring1 = 0
                    if sectionMapCount == 0: 
                        scoring += 0
                        scoring1 = 0
                    elif sectionMapCount == 1: 
                        scoring -= 1
                        scoring1 = -1 
                    elif sectionMapCount == 2: 
                        scoring -= 3
                        scoring1 = -3
                    else:
                        scoring -= 4 
                        scoring1 = -4   
                    
                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if standard_match_headings_count<=3:
                        scoring+=0
                        scoring1 = 0
                    else :
                        new_score = (standard_match_headings_count/actualHeadingsCount) * 5
                        scoring += new_score    
                        scoring1 = new_score       
                    print(scoring)
                    score_array.append(scoring1)


                    email_flag = 1 if len(email_finderSet) > 0 else 0
                    phone_flag = 1 if len(phone_all1) > 0 else 0

                    scoring1 = 0
                    if email_flag == 1:
                        scoring += 1
                        scoring1 = 1
                    else:
                        scoring -= 5
                        scoring1 = -5    
                

                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if phone_flag == 1:
                        scoring += 1
                        scoring1 = 1
                    else:
                        scoring -= 5
                        scoring1 = -5 
                
                    print(scoring)
                    score_array.append(scoring1)
            #changed
                    scoring1 = 0
                    if experience==0:

                        if Skills_Total >= 5 and Skills_Total < 18:
                            new_score =  ((Skills_Total)/(18))*9    
                            scoring += new_score 
                            scoring1 = new_score

                        elif Skills_Total >= 18 and Skills_Total <= 55:
                            scoring += 9
                            scoring1 = 9
                        elif Skills_Total > 55:
                            scoring += 4.5
                            scoring1 = 4.5
                        elif Skills_Total<=4:
                            scoring -= 6  
                            scoring1 = -6  
                        else:
                            scoring+=0
                            scoring1 = 0

                    elif experience<2:

                        if Skills_Total >= 5 and Skills_Total < 20:
                            new_score =  ((Skills_Total)/(20))*9    
                            scoring += new_score 
                            scoring1 = new_score

                        elif Skills_Total >= 20 and Skills_Total <= 55:
                            scoring += 9
                            scoring1 = 9

                        elif Skills_Total > 55:
                            scoring += 4.5
                            scoring1 = 4.5
                        elif Skills_Total<=4:
                            scoring -= 6  
                            scoring1 = -6  
                        else:
                            scoring+=0
                            scoring1 = 0

                    elif experience>=2 and experience<=5:

                        if Skills_Total >= 5 and Skills_Total < 24:
                            new_score =  ((Skills_Total)/(24))*9    
                            scoring += new_score 
                            scoring1 = new_score

                        elif Skills_Total >= 24 and Skills_Total <= 65:
                            scoring += 9
                            scoring1 = 9

                        elif Skills_Total > 65:
                            scoring += 4.5
                            scoring1 = 4.5
                        elif Skills_Total<=4:
                            scoring -= 6  
                            scoring1 = -6  
                        else:
                            scoring+=0
                            scoring1 = 0
                    elif experience>5:

                        if Skills_Total >= 5 and Skills_Total < 24:
                            new_score =  ((Skills_Total)/(24))*9    
                            scoring += new_score 
                            scoring1 = new_score

                        elif Skills_Total >= 24 and Skills_Total <= 65:
                            scoring += 9
                            scoring1 = 9

                        elif Skills_Total > 65:
                            scoring += 4.5
                            scoring1 = 4.5

                        elif Skills_Total<=4:
                            scoring -= 6  
                            scoring1 = -6  
                        else:
                            scoring+=0
                            scoring1 = 0
                    else:
                        scoring+=0
                        scoring1 = 0
                    print(scoring)
                    score_array.append(scoring1)
                    
                    scoring1 = 0 
                    if images =="No images found in the PDF file.":
                        scoring +=0
                        scoring1 = 0
                    else:
                        scoring -=4
                        scoring1 = -4
                    
                    print(scoring)
                    score_array.append(scoring1)



                    scoring1 = 0 
                    if len(repeated_words) > 3:
                        scoring -= 3
                        scoring1 = -3
                    elif len(repeated_words) >= 1 and len(repeated_words) <= 3:
                        scoring -= 1
                        scoring1 = -1
                    
                    else:
                        scoring += 3
                        scoring1 = 3
                    print(scoring)
                    score_array.append(scoring1)
                
                    


                    scoring1 = 0 
                    if match_strings(file_name, name_output):
                        scoring += 1
                        scoring1 = 1
                    else:
                        scoring += 0
                        scoring1 = 0
                    print(scoring)
                    print("Name")
                    score_array.append(scoring1)





                    scoring1 = 0 
                    if len(output_voice) > 2:
                        scoring -= 3
                        scoring1 = -3
                    elif len(output_voice) == 1 or len(output_voice) == 2:
                        scoring -= 1
                        scoring1 = -1
                    
                    else:
                        scoring += 3
                        scoring1 = 3
                    print(scoring)
                    score_array.append(scoring1)


                    scoring1 = 0 
                    if fillerwords_total > 3:
                        scoring -= 3
                        scoring1 = -3
                    elif fillerwords_total >= 1 and fillerwords_total <= 3:
                        scoring -= 1
                        scoring1 = -1
                    
                    else:
                        scoring += 2
                        scoring1 = 2
                    print(scoring)
                    score_array.append(scoring1)




                #changed
                    scoring1 = 0
                    if experience == 0:
                        if pages_count == 1 :
                            scoring += 5
                            scoring1 = 5
                        elif pages_count == 3 or pages_count == 2:
                            scoring -= 4
                            scoring1 = -4
                        else:
                            scoring -= 6 
                            scoring1 = -6  
                    
                    elif experience<2:
                        if pages_count == 1 or pages_count == 2:
                            scoring += 5
                            scoring1 = 5
                        elif pages_count == 3:
                            scoring -= 4
                            scoring1 = -4
                        else:
                            scoring -= 6 
                            scoring1 = -6  
                    elif experience>=2 and experience<=5:
                        if pages_count == 1 or pages_count == 2:
                            scoring += 5
                            scoring1 = 5
                        elif pages_count == 3:
                            scoring -= 4
                            scoring1 = -4
                        else:
                            scoring -= 6 
                            scoring1 = -6  
                    elif experience>5:
                        if pages_count == 1 or pages_count == 2:
                            scoring += 5
                            scoring1 = 5
                        elif pages_count == 3:
                            scoring -= 4
                            scoring1 = -4
                        else:
                            scoring -= 6 
                            scoring1 = -6  
                    else:
                        scoring+=0
                        scoring1 = 0
                    print(scoring)
                    score_array.append(scoring1)

                    personalPronouns_flag = 1 if len(personalPronouns)>0 else 0
                    scoring1 = 0
                    if personalPronouns_flag==1:
                        scoring -= 3
                        scoring1 = -3
                    else:
                        scoring += 0
                        scoring1 = 0
                    
                    print(scoring)
                    score_array.append(scoring1)

                    scoring1 = 0
                    if tables_flag==1:
                        scoring -= 4
                        scoring1 = -4
                    else:
                        scoring -= 0
                        scoring1 = 0
                
                    score_array.append(scoring1)
                    print(scoring)
                    print("net")
                
            # final scoring changed
                    scoring1 = 0
                    if scoring >=0 and scoring<10 :
                        if total_word_count>=0 and total_word_count<300:
                            scoring+=4
                            scoring1 = 4
                        else:
                            scoring+=7 
                            scoring1 = 7

                
            
                    if scoring >=10 and scoring<15 :
                        if total_word_count>=0 and total_word_count<300:
                            scoring+=2
                            scoring1 = 2
                        else:
                            scoring+=2  
                            scoring1 = 2    
                    


                    if scoring >=20 and scoring<25 and total_word_count>500:
                        scoring+=3
                        scoring1 = 3

                

                    if scoring >=25 and scoring<30 :
                        if total_word_count>=0 and total_word_count<300:
                            scoring-=4
                            scoring1 = -4
                        elif total_word_count>=300 and total_word_count<500:
                            scoring-=2
                            scoring1 = -2
                        else:
                            scoring+=0
                            scoring1 = 0

                

                    if scoring >=30 and scoring<35 :
                        if total_word_count>=0 and total_word_count<250:
                            scoring-=3
                            scoring1 = -3
                        elif total_word_count>=250 and total_word_count<400:
                            scoring+=0
                            scoring1 = 0
                        else:
                            scoring+=3
                            scoring1 = 3
                        
                    

                    if scoring >=35 and scoring<40 :
                        if total_word_count>=0 and total_word_count<275:
                            scoring-=8
                            scoring1 = -8
                        elif total_word_count>= 275 and total_word_count<400:
                            scoring-=3
                            scoring1 = -3
                        elif total_word_count>=400 and total_word_count<500:
                            scoring-=1.5
                            scoring1 = -1.5
                        else:
                            scoring+=0  
                            scoring1 = 0


                

                    if scoring >=45 and scoring<50 : 
                        if total_word_count>=0 and total_word_count<300:
                            scoring-=11
                            scoring1 = -11
                        elif total_word_count>=300 and total_word_count<400:
                            scoring-=6
                            scoring1 = -6
                        elif total_word_count>=400 and total_word_count<500:
                            scoring-=4
                            scoring1 = -4
                        else:
                            scoring+=0  
                            scoring1 = 0

                    if scoring >=50 and scoring<60: 
                        if total_word_count>=0 and total_word_count<300:
                            scoring-=5
                            scoring1 = -5
                        elif total_word_count>=300 and total_word_count<400:
                            scoring-=3
                            scoring1 = -3
                        else:
                            scoring+=0   
                            scoring1 = 0  
                    score_array.append(scoring1)
                    print(scoring)   
                    x = round(scoring)
                    return x,score_array



                except Exception as e:
                    print("Scoring ",e)
                    return 180,[]



        
        def get_chars(standard_font_style_flag,multiple_font_style,multiple_font_size,content_size_flag,font_colors_Total,standard_color_flag,actionwords_total,actionwords_total_negative,total_repeated_actionwords_negative,Bullets_Total,standard_bullet_flag,total_repeated_actionwords,ats_date,dates_nonAts,clean_measurable,total_word_count,email_finderSet,phonenumbers_finderSet,images,linkedIn_flag,url_flag,pages_count,personalPronouns,tables_flag,font_styles,font_sizes,max_size,actionwordsSet_negative,frequencyList_negative,phone_all1,skills,Work_Project_Headings,actionwordsSet,frequencyList,file_size_kb,headings,EduSkill_Headings,notRequired_Heading,Other_Headings,experience,file_name, name_output,output_voice,fillerwords):   
            try: 
                    print("✅ Inside get_chars function")
                    ans = [0]*26

                    mark = [0]*26

                    # font style  1
        
                    if standard_font_style_flag == 1:
                        if multiple_font_style==1:
                            ans[0] = f"Great Job! Detected <b>{multiple_font_style}</b> font style <b>{font_styles}</b> which is compliant with ATS standards"
                            mark[0] = 'tick'
                        elif multiple_font_style==2:
                            ans[0] = f"Detected <b>{multiple_font_style}</b> Font Styles <b>{font_styles}</b> and both of them are standard. As per the ATS Compliances, Use only 1 Standard Font Style"
                            mark[0] = 'tick'
                        else:
                            ans[0] = f"Detected <b>{multiple_font_style}</b> Font Styles <b>{font_styles}</b>, As per the ATS Compliances, Use only 1 Standard Font Style and not multiple"
                            mark[0] = 'exclamation'

                    if standard_font_style_flag ==0:
                        if multiple_font_style == 1:
                            ans[0] = f"Found <b>{multiple_font_style}</b> Font Style <b>{font_styles}</b> which is not complaint with ATS standards. Use standard font styles like Arial, Cambria, Georgia, Times New Roman etc."
                            mark[0] = 'cross'
                        elif multiple_font_style ==0:
                            ans[0]="No Font Styles Detected, seems like you have used rare ones. Use standard font styles like Arial, Cambria, Georgia, Times New Roman etc."                    
                            mark[0] = 'exclamation'                       
                        else:
                            ans[0] = f"Found <b>{multiple_font_style}</b> Font Styles <b>{font_styles}</b> and some of them are not standard as per ATS. It is advised to use only 1 ATS Friendly Font Style (Arial, Cambria, Georgia, Times New Roman etc.) for better readability"
                            mark[0] = 'exclamation'


                # font size  2
        
                    if multiple_font_size==1:
                        if content_size_flag==1:
                            ans[1] = f"Detected only <b>{multiple_font_size}</b> Font Size in the resume. Use different font sizes for content, section headings, & name as it makes it easy for ATS to read"
                            mark[1] = 'cross'
                        else:
                            ans[1] = f"Detected only <b>{multiple_font_size}</b> Font Size in the resume. Use different font sizes for content (preferably 10/11/12), section headings, & name as it makes it easy for ATS to read"
                            mark[1] = 'cross'

                    elif multiple_font_size == 2:
                        if content_size_flag==1:
                            ans[1] = f"Detected <b>{multiple_font_size}</b> different Font Sizes <b>{font_sizes}</b> with <b>{max_size}</b> as Content Size. Use different font sizes for content, section headings, & name as it makes it easy for ATS to read"
                            mark[1] = 'exclamation'
                        else:
                            ans[1] = f"Detected <b>{multiple_font_size}</b> different Font Sizes <b>{font_sizes}</b> with <b>{max_size}</b> as Content Size. Use different font sizes for content (preferably 10/11/12), section headings, & name as it makes it easy for ATS to read"
                            mark[1] = 'exclamation'
                    elif multiple_font_size == 3 or multiple_font_size == 4:
                        if content_size_flag==1:
                            ans[1] = f"Detected <b>{multiple_font_size}</b> different Font Sizes <b>{font_sizes}</b> with <b>{max_size}</b> as Content Size. Good Job with Font Size Optimization"
                            mark[1] = 'tick'
                        else:
                            ans[1] = f"Detected <b>{multiple_font_size}</b> different Font Sizes <b>{font_sizes}</b> with <b>{max_size}</b> as Content Size. Use Content Font Size as 10 , 11 or 12."
                            mark[1] = 'exclamation'
                    elif multiple_font_size==0:
                        if content_size_flag==0:
                            ans[1]="No Font Size Detected precisely, issue with readiability of the content. Use different font sizes for content, section headings, & name as it makes it easy for ATS to read"
                            mark[1] = 'cross'
                    else:
                        if content_size_flag==1:
                            ans[1] = f"Detected <b>{multiple_font_size}</b> different Font Sizes <b>{font_sizes}</b> with <b>{max_size}</b> as Content Size. Content Font Size is good, use of more than 3 Font Sizes should be avoided"
                            mark[1] = 'exclamation'
                        else:
                            ans[1] = f"Detected <b>{multiple_font_size}</b> different Font Sizes <b>{font_sizes}</b> with <b>{max_size}</b> as Content Size. Use Content Font Size as 10, 11 or 12 and avoid using of more than 3 font sizes."
                            mark[1] = 'exclamation'


                    # font colors  3
                    if font_colors_Total==0:
                        if standard_color_flag==0:
                            ans[2]="#NA"
                            mark[2] = '#NA'
                    elif font_colors_Total<=5:
                        if standard_color_flag==1:
                            ans[2] = f"<b>{font_colors_Total}</b> Font Colour were found in your resume. Good Job with that"
                            mark[2] = 'tick'

                        else:
                            ans[2] = f"<b>{font_colors_Total}</b> Font Colour were found in your resume and few of them are bright, bold or unwanted. Replace them with elegant Font Colours"
                            mark[2] = 'exclamation'
                    else:
                        if standard_color_flag==1:
                            ans[2] = f"<b>{font_colors_Total}</b> Font Colours were found in your resume. Limit the use of total font colours to less than 3"
                            mark[2] = 'cross'
                        
                        else:
                            ans[2] = f"Detected <b>{font_colors_Total}</b> Font Colours and few of them are bright, bold or unwanted. Use less than 3 Font Colours for professional presentation"
                            mark[2] = 'exclamation'

                    # action words 4
        
                       
                    str3 = ''
                    str3 = ', '.join(str(freq) for freq in frequencyList)
                    ac = list(actionwordsSet) 

        
                    if actionwords_total == 0:
                        if total_repeated_actionwords==0: 
                            ans[3] = f"Ohh, We could not detect any Action Words. Try to reiterate your bullet points using strong action words"
                            mark[3] = 'cross'
                    elif actionwords_total == 1:
                        if total_repeated_actionwords==0: 
                            ans[3] = f"Only <b>{actionwords_total}</b> Action word <b>{ac}</b> detected. Use more of these at the start of bullet points to increase the impact and catch recruiter's attention"
                            mark[3] = 'cross'
                        else:
                            ans[3] = f"Only <b>{actionwords_total}</b> Action word <b>{ac}</b> detected and is also being repeated. Increase the number of action words at the start of bullet points to increase the impact and catch recruiter's attention"
                            mark[3] = 'exclamation'
        
                    elif actionwords_total == 2 or actionwords_total == 3:
                        if total_repeated_actionwords==0: 
                            if len(Work_Project_Headings) == 0:
                                ans[3] = f"Only <b>{actionwords_total}</b> Action words <b>{ac}</b> are detected. Use more of these at the start of bullet points to increase the impact and catch recruiter's attention"
                                mark[3] = 'cross'
                            
                            else:
                                ans[3] = f"Only <b>{actionwords_total}</b> Action words <b>{ac}</b> are used in <b>{Work_Project_Headings}</b> section. Use more of these at the start of bullet points to increase the impact and catch recruiter's attention"
                                mark[3] = 'exclamation'
                        else:
                            if len(Work_Project_Headings) == 0:
                                ans[3] = f"Only <b>{actionwords_total}</b> Action words <b>{ac}</b> detected with <b>{total_repeated_actionwords}</b> being repeated multiple times <b>{str3}</b>. Use a variety of action words at the start of bullet points to catch recruiter's attention"
                                mark[3] = 'cross'
                            else:
                                ans[3] = f"Only <b>{actionwords_total}</b> Action words <b>{ac}</b> are used in <b>{Work_Project_Headings}</b> section with <b>{total_repeated_actionwords}</b> being repeated multiple times <b>{str3}</b>. Use a variety of action words at the start of bullet points to catch recruiter's attention" 
                                mark[3] = 'exclamation'
                
                    elif actionwords_total >= 4 and actionwords_total <= 8:
                        if total_repeated_actionwords==0: 
                            if len(Work_Project_Headings) > 0:
                                if actionwords_total == 4:
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found. Update your content in <b>{Work_Project_Headings}</b> and include more words to justify the impact of your work \n List of action words detected: <b>{ac}</b>"
                                    mark[3] = 'exclamation'
                                elif actionwords_total == 5:
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found. Update your content in <b>{Work_Project_Headings}</b> and include more words to justify the impact of your work \n List of action words detected: <b>{ac}</b>"
                                    mark[3] = 'exclamation'
                                else:
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found. Update your content in <b>{Work_Project_Headings}</b> and include more words to justify the impact of your work \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more"  
                                    mark[3] = 'exclamation'
                            else:
                                if actionwords_total == 4:   
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found. Update your content and include more words to justify the impact of your work \n List of action words detected: <b>{ac}</b>"  
                                    mark[3] = 'cross'
                                elif actionwords_total == 5: 
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found. Update your content and include more words to justify the impact of your work \n List of action words detected: <b>{ac}</b>"
                                    mark[3] = 'cross'
                                else: 
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found. Update your content and include more words to justify the impact of your work \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more" 
                                    mark[3] = 'cross'
                        else:
                            if len(Work_Project_Headings) > 0:    
                                if actionwords_total == 4: 
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found with <b>{total_repeated_actionwords}</b> being repeated multiple times. Use a variety of action words at the start of bullet points and avoid repetition \n List of action words detected: <b>{ac}</b> \n List of repeated action words detected: <b>{str3}</b>"
                                    mark[3] = 'exclamation'
                                elif actionwords_total == 5: 
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found with <b>{total_repeated_actionwords}</b> being repeated multiple times. Use a variety of action words at the start of bullet points and avoid repetition \n List of action words detected: <b>{ac[:5]}</b> \n List of repeated action words detected: <b>{str3}</b>" 
                                    mark[3] = 'exclamation'
                                else:   
                                    ans[3] = f"Only <b>{actionwords_total}</b> Actions words found with <b>{total_repeated_actionwords}</b> being repeated multiple times. Use a variety of action words at the start of bullet points and avoid repetition \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more \n List of repeated action words detected: <b>{str3}</b>"
                                    mark[3] = 'exclamation'
                                    
                                    
                            else:
                                if actionwords_total == 4: 
                                    ans[3] = f"Only <b>{actionwords_total}</b> Action words found with <b>{total_repeated_actionwords}</b> being repeated multiple times. Use a variety of action words at the start of bullet points and avoid repetition \n List of action words detected: <b>{ac}</b> \n List of repeated action words detected: <b>{str3}</b>"
                                    mark[3] = 'exclamation'
                                elif actionwords_total == 5: 
                                    ans[3] = f"Only <b>{actionwords_total}</b> Action words found with <b>{total_repeated_actionwords}</b> being repeated multiple times. Use a variety of action words at the start of bullet points and avoid repetition \n List of action words detected: <b>{ac[:5]}</b> \n List of repeated action words detected: <b>{str3}</b>" 
                                    mark[3] = 'exclamation'
                                else:   
                                    ans[3] = f"Only <b>{actionwords_total}</b> Action words found with <b>{total_repeated_actionwords}</b> being repeated multiple times. Use a variety of action words at the start of bullet points and avoid repetition \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more \n List of repeated action words detected: <b>{str3}</b>"
                                    mark[3] = 'exclamation'
                    else:
                        if total_repeated_actionwords==0: 
                            if len(Work_Project_Headings) == 0:
                                ans[3] = f"Good Job, Detected <b>{actionwords_total}</b> action words to justify your impact of work \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more"
                                mark[3] = 'tick'
                            else:
                                ans[3] = f"Good Job, Detected <b>{actionwords_total}</b> action words to justify your impact of work in <b>{Work_Project_Headings}</b>.\n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more" 
                                mark[3] = 'tick'
                        else:
                            if len(Work_Project_Headings) == 0:
                                if total_repeated_actionwords == 1:     
                                    ans[3] = f"Good Job, Detected <b>{actionwords_total}</b> action words to justify your impact of work but <b>'{total_repeated_actionwords}'</b>, being repeated multiple times \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more \n List of repeated action words detected: <b>{str3}</b>" 
                                    mark[3] = 'exclamation'
                                else:
                                    ans[3] = f"Good Job, Detected <b>{actionwords_total}</b> action words to justify your impact of work but <b>'{total_repeated_actionwords}'</b>,  being repeated multiple times \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more \n List of repeated action words detected: <b>{str3}</b>" 
                                    mark[3] = 'exclamation'
                            else:
                                if total_repeated_actionwords == 1:     
                                    ans[3] = f"Good Job, Detected <b>{actionwords_total}</b> action words to justify your impact of work in <b>{Work_Project_Headings}</b> section but <b>'{total_repeated_actionwords}'</b>, being repeated multiple times \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more \n List of repeated action words detected: <b>{str3}</b>"    
                                    mark[3] = 'exclamation'
                                else:
                                    ans[3] = f"Good Job, Detected <b>{actionwords_total}</b> action words to justify your impact of work in <b>{Work_Project_Headings}</b> section but <b>'{total_repeated_actionwords}'</b>, being repeated multiple times \n List of action words detected: <b>{ac[:5]}</b> and  <b>{len(ac) - len(ac[:5])}</b> more \n List of repeated action words detected: <b>{str3}</b>"    
                                    mark[3] = 'exclamation'







                    
                    # action words negative 5
        
            
                    str1 = '' 
                    str1 = ', '.join(str(freq) for freq in  frequencyList_negative)
                    if actionwords_total_negative == 0:
                        ans[4] = "Awesome! No Negative Action Words or Buzzwords found in your Resume"
                        mark[4] = 'tick'
                    else:  
                        if total_repeated_actionwords_negative==0:
                            ans[4] = f"<b>{actionwords_total_negative}</b> Negative Action Words found <b>{actionwordsSet_negative}</b>. Avoid using these to avoid red flags for recruiters"
                            mark[4] = 'cross'
                        else:
                            if total_repeated_actionwords_negative == 1:    
                                ans[4] = f"<b>{actionwords_total_negative}</b> Negative Action Words found <b>{actionwordsSet_negative}</b> with <b>{total_repeated_actionwords_negative}</b> word <b>{str1}</b> being repated multiple times. Avoid using these to avoid red flags for recruiters"
                                mark[4] = 'cross'
                            else:
                                ans[4] = f"<b>{actionwords_total_negative}</b> Negative Action Words found <b>{actionwordsSet_negative}</b> with <b>{total_repeated_actionwords_negative}</b> words <b>{str1}</b> being repated multiple times. Avoid using these to avoid red flags for recruiters"
                                mark[4] = 'cross'


                    # headings 6
            
                    if len(subHeadings):
                        ans[5] = f"Following Sections Found in your Resume:\n <b>{subHeadings}</b>"
                        mark[5] = 'tick'
                    else:
                        ans[5] = "#NA"
                        mark[5] = '#NA'

                    # edu skill headings 7
        
                    if len(EduSkill_Headings)==0:
                        ans[6] = f'"Skills"/"Education" Sections are important which is missing' 
                        mark[6] = 'cross'
                    elif len(EduSkill_Headings)==1:
                        if EduSkill_Headings[0]=="Education":
                            ans[6] = f'"Skills" Section is important which is missing'
                            mark[6] = 'exclamation'
                        else:
                            ans[6] = f'"Education" Section is important which is missing'  
                            mark[6] = 'exclamation'
                    else:
                        ans[6] = "#NA"   
                        mark[6] = '#NA'
                    # 8
        
                    if len(notRequired_Heading):
                        ans[7] = f'Following Sections are not Required in the Resume <b>{notRequired_Heading}</b>. Avoid adding these sections'
                        mark[7] = 'cross' 
                    else:
                        ans[7] = "#NA"
                        mark[7] = '#NA'
        
                    # other relevant section headings 9   
                    if len(Other_Headings)>1:
            
                        ans[8] = f'"Good Job, <b>{len(Other_Headings)}</b> Addtional sections are mentioned which adds on to your overall candidature:\n <b>{Other_Headings}</b>'
                        mark[8] = 'tick'
                    elif len(Other_Headings)==1:
                        ans[8] = f'"Good Job, <b>{len(Other_Headings)}</b> Addtional section is mentioned which adds on to your overall candidature:\n <b>{Other_Headings}</b>'
                        mark[8] = 'tick'  
                    else:
                        ans[8] = "#NA"
                        mark[8] = '#NA'
            # bullets 10
        
                    if Bullets_Total==1:
                        if standard_bullet_flag==1:
                            ans[9] = "Great job with the use of bullets, all of them are ATS Compliant"
                            mark[9] = 'tick' 
                        else:
                            ans[9] = "Change your current bullets with ATS standard ones"
                            mark[9] = 'exclamation'
                    elif Bullets_Total>1:
                        if standard_bullet_flag==1:
                           ans[9] = f"Found <b>{Bullets_Total}</b> different bullets. Only single bullet type should be used for consistency"
                           mark[9] = 'exclamation'
                        else:
                            ans[9] = f"Found <b>{Bullets_Total}</b> bullets and all of them are <b>Non ATS Friendly</b>. Only single bullet type should be used for consistent content readability"
                            mark[9] = 'cross'
                    else:
                        ans[9] = "We could not detect any Bullets. Use standard bullets to explain different sections of your resume. In case you have used the bullets.Its not readable by ATS"
                        mark[9] = 'cross'  
            # dates type 11
            
                    ats_length = 1 if len(ats_date) > 0 else 0
                    nonats_length = 1 if len(dates_nonAts) > 0 else 0
            
            
                    if ats_length==1:
                        if nonats_length==1:
                            ans[10] = f"Detected both ATS and Non-ATS Dates Format. Replace Non-ATS formats <b>{dates_nonAts}</b> with standard ones."
                            mark[10] = 'exclamation'
                        else:
                            ans[10] = f"Great Work, All the dates used are ATS Compliant"
                            mark[10] = 'tick'
                    else:
                        if nonats_length==1:
                            ans[10] = f'All the dates used in the resume are not ATS compliant. Consider removing these: <b>{dates_nonAts}</b> Refer to a few ATS Dates format: "MM/YYY OR MON YYY - MON YYYY"'
                            mark[10] = 'cross'
                        else:
                            ans[10] = f"No Dates were detected in the resume. Consider adding dates in all these section: Work Experience, Education, Projects etc.."
                            mark[10] = 'cross'
            # measurable count 12
        
                    total_measurable=len(clean_measurable)
                    mes = list(clean_measurable)
                    if total_measurable == 0:
                        ans[11] = f"Ohh, We could detect no quantifiable accomplishment in your resume. Add more bullet points that use numerical metrics to highlight your contributions"
                        mark[11] = 'cross'
                    elif total_measurable == 1:
                        ans[11] = f"We could detect only 1 quantifiable accomplishment, which quite low. Add more bullet points that use numerical metrics to highlight your contributions"
                        mark[11] = 'cross'
                    elif total_measurable >= 2 and total_measurable<4:
                        ans[11] = f"We could detect only about <b>{total_measurable}</b> quantifiable accomplishments, which is relatively low. Add more bullet points that use numerical metrics to highlight your contributions"
                        mark[11] = 'cross'
                    elif total_measurable >= 4 and total_measurable <= 8:
                        if total_measurable == 4: 
                            ans[11] = f"We could only detect <b>{total_measurable}</b> quantifiable accomplishments, which is relatively low. Add a few more numerical metrics to highlight your contributions \n List of quantifiable achievements: <b>{clean_measurable}</b>"
                            mark[11] = 'exclamation'
                        elif total_measurable == 5:
                            ans[11] = f"We could only detect <b>{total_measurable}</b> quantifiable accomplishments, which is relatively low. Add a few more numerical metrics to highlight your contributions \n List of quantifiable achievements: <b>{mes[:5]}</b>"
                            mark[11] = 'exclamation'
                        else:
                            ans[11] = f"We could only detect <b>{total_measurable}</b> quantifiable accomplishments, which is relatively low. Add a few more numerical metrics to highlight your contributions \n List of quantifiable achievements: <b>{mes[:5]}</b> and <b>{len(mes) - len(mes[:5])}</b> more"
                            mark[11] = 'exclamation'
                    else:
                        ans[11] = f"Perfect, Your resume mentions around <b>{total_measurable}</b> measurable achievements. \n List of quantifiable achievements: <b>{mes[:5]}</b> and <b>{len(mes) - len(mes[:5])}</b>  more"          
                        mark[11] = 'tick'
                  
                  
                    #word count 13 
                    if experience==0:
                        if total_word_count<150:
                            ans[12] = f"Your Resume has only <b>{total_word_count}</b> Words. You should increase the word count to 300+ atleast"
                            mark[12] = 'cross'            
                        elif total_word_count>=150 and total_word_count<=165:
                            ans[12] = f"Your Resume has quite a few words  <b>{total_word_count}</b>. You should increase the word count to atleast 300+ atleast"
                            mark[12] = 'exclamation' 
                        elif total_word_count>165 and total_word_count<=300:
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Most successful resumes have word count between 300 to 450, consider increasing your word count by adding relevant sections"
                            mark[12] = 'exclamation' 
                        elif total_word_count>300 and total_word_count<=450:
                            ans[12] = f"Perfect, Your resume contains <b>{total_word_count}</b> words. Most successful resume have word count in this range"
                            mark[12] = 'tick' 
                        elif total_word_count>450 and total_word_count<=600:
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Consider deleting irrelevant sections and bullet points"
                            mark[12] = 'exclamation' 
                        else: 
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Most successful resumes have word count between 400 to 600, consider deleting irrelevant sections and bullet points"
                            mark[12] = 'cross'  
                    elif experience<2:
                        if total_word_count<=165:
                            ans[12] = f"Your Resume has only <b>{total_word_count}</b> Words. You should increase the word count to 350+ atleast"
                            mark[12] = 'cross' 
                        elif total_word_count>165 and total_word_count<=250:
                            ans[12] = f"Your Resume has quite a few words  <b>{total_word_count}</b>. You should increase the word count to atleast 350+ atleast"
                            mark[12] = 'exclamation' 
                        elif total_word_count>250 and total_word_count<=350:
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Most successful resumes have word count between 350 to 550, consider increasing your word count by adding relevant sections"
                            mark[12] = 'exclamation' 
                        elif total_word_count>350 and total_word_count<=650:
                            ans[12] = f"Perfect, Your resume contains <b>{total_word_count}</b> words. Most successful resume have word count in this range"
                            mark[12] = 'tick' 
                        elif total_word_count>650 and total_word_count<=850:
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Consider deleting irrelevant sections and bullet points"
                            mark[12] = 'exclamation' 
                        else: 
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Most successful resumes have word count between 350 to 650, consider deleting irrelevant sections and bullet points"
                            mark[12] = 'cross' 
                    elif experience>=2 and experience<=5:
                        if total_word_count<=165:
                            ans[12] = f"Your Resume has only <b>{total_word_count}</b> Words. You should increase the word count to 350+ atleast"
                            mark[12] = 'cross' 
                        elif total_word_count>165 and total_word_count<=250:
                            ans[12] = f"Your Resume has quite a few words  <b>{total_word_count}</b>. You should increase the word count to atleast 350+ atleast"
                            mark[12] = 'exclamation' 
                        elif total_word_count>250 and total_word_count<=350:
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Most successful resumes have word count between 350 to 650, consider increasing your word count by adding relevant sections"
                            mark[12] = 'exclamation' 
                        elif total_word_count>350 and total_word_count<=650:
                            ans[12] = f"Perfect, Your resume contains <b>{total_word_count}</b> words. Most successful resume have word count in this range"
                            mark[12] = 'tick' 
                        elif total_word_count>650 and total_word_count<=850:
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Consider deleting irrelevant sections and bullet points"
                            mark[12] = 'exclamation'  
                        else: 
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Most successful resumes have word count between 350 to 650, consider deleting irrelevant sections and bullet points"
                            mark[12] = 'cross' 
                    else:
                        if total_word_count<165:
                            ans[12] = f"Your Resume has only <b>{total_word_count}</b> Words. You should increase the word count to 400+ atleast"
                            mark[12] = 'cross' 
                        elif total_word_count>=165 and total_word_count<=200:
                            ans[12] = f"Your Resume has quite a few words  <b>{total_word_count}</b>. You should increase the word count to atleast 400+ atleast"
                            mark[12] = 'exclamation' 
                        elif total_word_count>200 and total_word_count<=400:
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Most successful resumes have word count between 400 to 800, consider increasing your word count by adding relevant sections"
                            mark[12] = 'exclamation' 
                        elif total_word_count>400 and total_word_count<=850:
                            ans[12] = f"Perfect, Your resume contains <b>{total_word_count}</b> words. Most successful resume have word count in this range"
                            mark[12] = 'tick' 
                        elif total_word_count>850 and total_word_count<=1000:
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Consider deleting irrelevant sections and bullet points"
                            mark[12] = 'exclamation' 
                        else: 
                            ans[12] = f"Your resume contains <b>{total_word_count}</b> words. Most successful resumes have word count between 400 to 800, consider deleting irrelevant sections and bullet points"
                            mark[12] = 'cross' 



                    email_flag = 1 if len(email_finderSet) > 0 else 0
                    phone_flag=1 if len(phone_all1) > 0 else 0
                    linkedin_flag=linkedIn_flag

            # email 14
        
                    if email_flag==1:
                        ans[13] = f"Email Detected <b>{email_finderSet}</b>"
                        mark[13] = 'tick' 
                    else:
                        ans[13] = "Your Email-ID is <b>missing</b>. How will recruiters reach out to you?"
                        mark[13] = 'cross' 
            # phone 15
        
                    if phone_flag==1:
                        ans[14] = f"Phone Number Detected <b>{phone_all1}</b>"
                        mark[14] = 'tick' 
                    else:
                        ans[14] = "Phone Number is <b>missing</b>. Add your contact details (Phone) in the resume for recruiters to reach you"
                        mark[14] = 'cross' 
            # linkedIn 16

                    if linkedin_flag==1:
                        ans[15] = "Linkedin Detected. Good Job, Optimise your Linkedin for better job opportunities"
                        mark[15] = 'tick' 
                    else:
                        ans[15] = "Linkedin is either <b>missing or hyperlinked</b>. Make sure to add your linkedin (complete URL) for recruiters to verify your profile"
                        mark[15] = 'cross' 

                    
            # total skills 17
        
                    str2 = []
                    str2  = [str(skill).upper() for skill in skills]

                    if experience==0:

                        if Skills_Total == 0:
                            ans[16] = f"Damn, We could see that you haven't used any Keyword. Keywords are most important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'cross' 
                        elif Skills_Total == 1:
                            ans[16] = f"Only 1 Keyword detected in your resume <b>{str2}</b>. Keywords are very important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'cross' 
                        elif Skills_Total > 1 and Skills_Total <= 4:
                            ans[16] = f"Only <b>{Skills_Total}</b> Keywords present in your resume <b>{str2}</b>. Keywords are very important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'exclamation' 
                        
                        elif  Skills_Total >= 5 and Skills_Total < 18:
                            if Skills_Total == 5: 
                                ans[16] = f"Quite a few Keywords <b>{Skills_Total}</b> are present in your resume.  Add a few more to increase your chances of interview calls by 50% \n List of keywords detected: <b>{str2[:5]}</b>"
                                mark[16] = 'exclamation' 
                            else:
                                ans[16] = f"Quite a few Keywords <b>{Skills_Total}</b> are present in your resume.  Add a few more to increase your chances of interview calls by 50% \n List of keywords detected: <b>{str2[:5]}</b> and <b>{len(str2) - len(str2[:5])}</b>  more"    
                                mark[16] = 'exclamation'  
                        elif  Skills_Total >= 18 and Skills_Total <= 55:
                            ans[16] = f"Great Job, <b>{Skills_Total}</b> Keywords detected in your resume. Keywords help you bypass ATS and justify your interest in a particular profile \n List of keywords detected: <b>{str2[:5]}</b> and <b>{len(str2) - len(str2[:5])}</b> more" 
                            mark[16] = 'tick'
                        else:
                            ans[16] = f"Damn! <b>{Skills_Total}</b> Keywords detected in your resume. Having more than 50+ keywords display less clarity towards a specific profile. Limit your Keywords to a particular Job Profile (<50)"
                            mark[16] = 'cross'

                    elif experience<2:

                        if Skills_Total == 0:
                            ans[16] = f"Damn, We could see that you haven't used any Keyword. Keywords are most important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'cross' 
                        elif Skills_Total == 1:
                            ans[16] = f"Only 1 Keyword detected in your resume <b>{str2}</b>. Keywords are very important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'cross'
                        elif Skills_Total > 1 and Skills_Total <= 4:
                            ans[16] = f"Only <b>{Skills_Total}</b> Keywords present in your resume <b>{str2}</b>. Keywords are very important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'exclamation'
                        elif  Skills_Total >= 5 and Skills_Total < 20:
                            if Skills_Total == 5: 
                                ans[16] = f"Quite a few Keywords <b>{Skills_Total}</b> are present in your resume.  Add a few more to increase your chances of interview calls by 50% \n List of keywords detected: <b>{str2[:5]}</b>"
                                mark[16] = 'exclamation'
                            else:
                                ans[16] = f"Quite a few Keywords <b>{Skills_Total}</b> are present in your resume.  Add a few more to increase your chances of interview calls by 50% \n List of keywords detected: <b>{str2[:5]}</b> and <b>{len(str2) - len(str2[:5])}</b>  more"    
                                mark[16] = 'exclamation'
                        elif  Skills_Total >= 20 and Skills_Total <= 55:
                            ans[16] = f"Great Job, <b>{Skills_Total}</b> Keywords detected in your resume. Keywords help you bypass ATS and justify your interest in a particular profile \n List of keywords detected: <b>{str2[:5]}</b> and <b>{len(str2) - len(str2[:5])}</b> more" 
                            mark[16] = 'tick'
                        else:
                            ans[16] = f"Damn! <b>{Skills_Total}</b> Keywords detected in your resume. Having more than 50+ keywords display less clarity towards a specific profile. Limit your Keywords to a particular Job Profile (<50)"
                            mark[16] = 'cross'
                    elif experience>=2 and experience<=5:

                        if Skills_Total == 0:
                            ans[16] = f"Damn, We could see that you haven't used any Keyword. Keywords are most important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'cross'
                        elif Skills_Total == 1:
                            ans[16] = f"Only 1 Keyword detected in your resume <b>{str2}</b>. Keywords are very important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'cross'
                        elif Skills_Total > 1 and Skills_Total <= 4:
                            ans[16] = f"Only <b>{Skills_Total}</b> Keywords present in your resume <b>{str2}</b>. Keywords are very important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'exclamation'
                        elif  Skills_Total >= 5 and Skills_Total < 24:
                            if Skills_Total == 5: 
                                ans[16] = f"Quite a few Keywords <b>{Skills_Total}</b> are present in your resume.  Add a few more to increase your chances of interview calls by 50% \n List of keywords detected: <b>{str2[:5]}</b>"
                                mark[16] = 'exclamation'
                            else:
                                ans[16] = f"Quite a few Keywords <b>{Skills_Total}</b> are present in your resume.  Add a few more to increase your chances of interview calls by 50% \n List of keywords detected: <b>{str2[:5]}</b> and <b>{len(str2) - len(str2[:5])}</b>  more"    
                                mark[16] = 'exclamation'
                        elif  Skills_Total >= 24 and Skills_Total <= 65:
                            ans[16] = f"Great Job, <b>{Skills_Total}</b> Keywords detected in your resume. Keywords help you bypass ATS and justify your interest in a particular profile \n List of keywords detected: <b>{str2[:5]}</b> and <b>{len(str2) - len(str2[:5])}</b> more" 
                            mark[16] = 'tick'
                        else:
                            ans[16] = f"Damn! <b>{Skills_Total}</b> Keywords detected in your resume. Having more than 50+ keywords display less clarity towards a specific profile. Limit your Keywords to a particular Job Profile (<50)"
                            mark[16] = 'cross'
                    else:

                        if Skills_Total == 0:
                            ans[16] = f"Damn, We could see that you haven't used any Keyword. Keywords are most important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'cross'
                        elif Skills_Total == 1:
                            ans[16] = f"Only 1 Keyword detected in your resume <b>{str2}.</b> Keywords are very important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'cross'
                        elif Skills_Total > 1 and Skills_Total <= 4:
                            ans[16] = f"Only <b>{Skills_Total}</b> Keywords present in your resume <b>{str2}.</b> Keywords are very important to bypass ATS. Add more Keywords as per your Profile"
                            mark[16] = 'exclamation'
                        elif  Skills_Total >= 5 and Skills_Total < 24:
                            if Skills_Total == 5: 
                                ans[16] = f"Quite a few Keywords <b>{Skills_Total}</b> are present in your resume.  Add a few more to increase your chances of interview calls by 50% \n List of keywords detected: <b>{str2[:5]}</b>"
                                mark[16] = 'exclamation'
                            else:
                                ans[16] = f"Quite a few Keywords <b>{Skills_Total}</b> are present in your resume.  Add a few more to increase your chances of interview calls by 50% \n List of keywords detected: <b>{str2[:5]}</b> and <b>{len(str2) - len(str2[:5])}</b>  more "    
                                mark[16] = 'exclamation' 
                        elif  Skills_Total >= 24 and Skills_Total <= 65:
                            ans[16] = f"Great Job, <b>{Skills_Total}</b> Keywords detected in your resume. Keywords help you bypass ATS and justify your interest in a particular profile \n List of keywords detected: <b>{str2[:5]}</b> and <b>{len(str2) - len(str2[:5])}</b> more " 
                            mark[16] = 'tick'
                        else:
                            ans[16] = f"Damn! <b>{Skills_Total}</b> Keywords detected in your resume. Having more than 50+ keywords display less clarity towards a specific profile. Limit your Keywords to a particular Job Profile (<50)"
                            mark[16] = 'cross'
            # images 18
        
                    if images=="Human Image Not Detected":
                     ans[17] = "We detected Images/Graphics in your Resume. Graphics hampers with ATS and should be avoided"
                     mark[17] = 'cross'
                    elif images=="Human Image Detected":
                     ans[17] = "We detected your photo in the Resume. Photograph hampers with ATS and is not required"
                     mark[17] = 'cross' 
                    else:
                        ans[17] = "#NA"
                        mark[17] = '#NA'
            # page count 19
        
                    if experience==0:

                        if pages_count==1 :
                            ans[18] = "Your resume meets the standard guidelines for number of pages"
                            mark[18] = 'tick'
                        elif pages_count == 3 or pages_count==2:
                            ans[18] = "Try to Reduce your Resume to 1 Page if possible"
                            mark[18] = 'exclamation'
                        elif pages_count > 3 :
                            ans[18] = "Your Resume is in <b>more than 3 pages</b>. Reduce it to less than 1 page since a recruiter only takes 6 seconds to scan your resume"
                            mark[18] = 'cross'
                        else:
                            ans[18]="#NA"
                            mark[18] = '#NA'
                    if experience<2:

                        if pages_count==1 :
                            ans[18] = "Your resume meets the standard guidelines for number of pages"
                            mark[18] = 'tick'
                        elif pages_count == 3 or pages_count==2:
                            ans[18] = "Try to Reduce your Resume to 1 Page if possible"
                            mark[18] = 'exclamation' 
                        elif pages_count > 3:
                            ans[18] = "Your Resume is in <b>3+ pages</b>. Reduce it to 1 page since recruiters onlt takes 6 seconds to scan your resume"
                            mark[18] = 'cross'
                        else:
                            ans[18]="#NA"
                            mark[18] = '#NA'
                    if experience>=2 and experience<=5:

                        if pages_count==1 or pages_count==2:
                            ans[18] = "Your resume meets the standard guidelines for number of pages"
                            mark[18] = 'tick'
                        elif pages_count >= 3:
                            ans[18] = "Your Resume is in <b>3 or more pages</b>. Reduce it to 1 page since recruiters onlt takes 6 seconds to scan your resume"
                            mark[18] = 'exclamation'
                        # elif pages_count > 3:
                        #     ans[18] = "Your Resume is in more than 3 pages. Reduce it to less than 2 pages since a recruiter only takes 6 seconds to scan your resume"
                        else:
                            ans[18]="#NA"
                            mark[18] = '#NA'

                    else:

                        if pages_count==1 or pages_count==2:
                            ans[18] = "Your resume meets the standard guidelines for number of pages"
                            mark[18] = 'tick'
                        elif pages_count >= 3:
                            ans[18] = "Your Resume is in <b>3 or more pages,/b>. Reduce it to 1 page since recruiters onlt takes 6 seconds to scan your resume"
                            mark[18] = 'exclamation'
                        # elif pages_count > 3:
                        #     ans[18] = "Your Resume is in more than 3 pages. Reduce it to less than 2 pages since a recruiter only takes 6 seconds to scan your resume"
                        else:
                            ans[18]="#NA"
                            mark[18] = '#NA'


            # personal pronouns 20
        
                    personalPronouns_flag = 0 if len(personalPronouns)>0 else 1

                    personalPronouns = set(personalPronouns)
                    personalPronouns = list(personalPronouns)

                    # result = '"' + ','.join(personalPronouns) + '"'

                    

                    if personalPronouns_flag==0:
                        ans[19] = f"Detected these: <b>{personalPronouns}</b>  personal pronouns on your resume. Ideally personal pronouns should not be used in any section of a resume"
                        mark[19] = 'cross'
                    else:
                        ans[19] = '#NA'
                        mark[19] = '#NA'
            # tables 21

                    if tables_flag==1:
                        ans[20] = "We found Tables in your resume which are not recognized by ATS and messes up your information. Try to avoid the use of tables"
                        mark[20] = 'cross'
                    else:
                        ans[20] = "#NA"
                        mark[20] = '#NA'

            # file size 22
            
                    file_size_kb = round(file_size_kb)
                    if file_size_kb  <= 1500:   
                        ans[21] = f"The file size of your resume is: <b>{file_size_kb}</b> KB \n This is compact enough to prevent any problems while submitting job applications."
                        mark[21] = 'tick'
                    else:
                        ans[21] = f"The file size of your resume is: <b>{file_size_kb}</b> KB \n Reduce it to less than 1500 KB to prevent any problems while submitting job applications"
                        mark[21] = 'cross' 
        
                    
            #  name 23
                    if match_strings(file_name, name_output):
                         ans[22] = 'Your file name seems to be synchornised as that of your name. Best format for file name should be "Your Name_Profile"'
                         mark[22] = 'tick'
                    else:
                          ans[22] = 'Damn, We detected your file name to be in a wrong format. It should be renamed as "Your Name_Profile"'

                          mark[22] = 'cross'

            

            #  Active Voice 24
                    voice_total = len(output_voice)  
                    # print(output_voice)
                    # print(voice_total)    
                    print("comments Working")
                    if voice_total == 0:
                         ans[23] = "Good Job, No Passvie Voice Instance Found"
                         mark[23] = 'tick'
                    else:
                          ans[23] = f"Found <b>{voice_total}</b> instance in Passive voice in your Resume. Re-write these in active voice to make a clear impact: \n Passive Voices Detected: <b>{output_voice}</b>"
               
                          mark[23] = 'cross'
            


             #  fillerWords 25
                           
                    word_counts = Counter(fillerwords)

                    fillerwords_freq = [word for word, count in word_counts.items() if count >= 2]
                    fillerwords_freq_total = len(fillerwords_freq)  
                    fillerwords_total = len(fillerwords)      
                    if fillerwords_total == 0:
                         ans[24] = "Awesome! No Filler Words found in your Resume"
                         mark[24] = 'tick'

                    elif fillerwords_total == 1:
                         ans[24] = f"<b>{fillerwords_total}</b> Filler Words found <b>{fillerwords}</b>. Remove these or Replace with Quantificable Impact / Numbers"
                         mark[24] = 'cross'

                    else:
                         if fillerwords_freq_total == 0:
                            ans[24] = f"<b>{fillerwords_total}</b> Filler Words found <b>{fillerwords}</b> with no words being repeated multiple times. Avoid using these to avoid red flags for recruiters"

                            mark[24] = 'cross'

                         elif fillerwords_freq_total == 1:
                            ans[24] = f"<b>{fillerwords_total}</b> Filler Words found <b>{fillerwords}</b> with <b>{fillerwords_freq}</b> word <b>{fillerwords_freq_total}</b> being repeated multiple times. Avoid using these to avoid red flags for recruiters"

                            mark[24] = 'cross'    
                         else: 
                            ans[24] = f"<b>{fillerwords_total}</b> Filler Words found <b>{fillerwords}</b> with <b>{fillerwords_freq}</b> words <b>{fillerwords_freq_total}</b> being repeated multiple times. Avoid using these to avoid red flags for recruiters"

                            mark[24] = 'cross'




                    return ans,mark
            except Exception as e:
                    print("Comments ",e)
                    ans = []
                    mark = []
                    return ans,mark
            

        def get_commentList(repeated_words):
            try:
                 print("Inside get_commentList")
                 print("")
                 
                 ans = [0]*1
                 
                 mark = [0]*1

                 if len(repeated_words) > 0:
                       ans1 = []
                     

                       for key, value in repeated_words.items():
                        ans1.append(f"Damn, <b>{len(repeated_words)}</b> Repeated Phrases Detected. \n Repeated Phrase: <b>{key}  ({value})</b>")
                         
                       mark[0] = 'cross'

                 else:
                      ans1 = []       

                 ans[0] = ans1
                 return ans,mark
            except Exception as e:
                 
                    print("Comments List",e)
                    ans = []
                    mark = []
                    return ans,mark


        print("\n📂 Reading File")   
        # print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        # s3 = boto3.client('s3',
        #     aws_access_key_id=aws_access_key,
        #     aws_secret_access_key=aws_secret_key,
        #     region_name=aws_region
        #     )
        # pdf_file = s3.get_object(Bucket=aws_s3_bucket_name, Key=key)['Body'].read()
        # pdf_file2 = s3.get_object(Bucket=aws_s3_bucket_name, Key=key)
        
        # UNCOMMENT TO RUN LOCALLY
        # ------------------------------------------------------------------------------------------------
        # read file locally

        # with open(key, "rb") as f:
        #     pdf_file = f.read()
        
        # Mimic what S3 would return
        pdf_file2 = {
            "ContentLength": 500,
            "ContentType": "application/pdf"
        }
        # ------------------------------------------------------------------------------------------------
        print("✅ File Read")
        # print("hey")
        # print(pdf_file)

        pdf_file_path = key
    
        # Read the PDF file as bytes
        with open(pdf_file_path, 'rb') as f:
            pdf_file = f.read()
        # pdf_file = key
        text = pdfText(key)

        # print(len(text),"length pf text for file",key)
        print(f"📏 Length of pdf: {len(text)} characters")
        if len(text) < 40:
            dta={
            "features":"false",
            "key":key,
            "wordLength":len(text)
            }
            headers = {"Content-Type": "application/json"}
            data = json.dumps(dta).encode('utf-8')
            try:
                response2 = requests.post(f"{root_url}/features/lambdaResponse", data=data, headers=headers, timeout=10)
            except requests.exceptions.ReadTimeout:
            
                print("overrrrrrrrrrrr text length less than 20")
                response2 = None
                response = {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "OPTIONS,POST,GET"  # Adjust the allowed HTTP methods as needed
                },
                "body": json.dumps(dta)
                }
                print(response)

                return response

        try:

            # print("at 1")
            font_styles,standard_font_style_flag,multiple_font_style,font_sizes,multiple_font_size = get_font_style_size(pdf_file)
            print("🔍 get_font_style_size() outputs: ")
            # print(f"\n{font_styles} \n{standard_font_style_flag} \n{multiple_font_style} \n{font_sizes} \n{multiple_font_size}\n\n")
            print(f"\nFont Styles: {font_styles}, \nStandard Font Style: {standard_font_style_flag}, \nMultiple Font Styles: {multiple_font_style}, \nFont Sizes: {font_sizes}, \nMultiple Font Sizes: {multiple_font_size}\n\n")
            # print("at 2")
            
            max_size,content_size_flag = get_maxSize_words(pdf_file)
            print("🔍 get_maxSize_words() outputs: ")
            # print("at 3")
            print(f"\nMax Size: {max_size}, \nContent Size: {content_size_flag}\n\n")
            
            
            font_colors,font_colors_Total,standard_color_flag= get_font_color(pdf_file)
            print("🔍 get_font_color() outputs: ")
            print(f"\nFont Colors: {font_colors}, \nFont Colors Total: {font_colors_Total}, \nStandard Color: {standard_color_flag}\n\n")
            # print("at 4")


            headings,subHeadings,notRequired_Heading,Work_Project_Headings,EduSkill_Headings,Other_Headings,Other_headings_db,sectionMap,sectionMapCount,standard_match_headings,standard_match_headings_count =  get_headings(pdf_file)
            # print(headings)
            print("🔍 get_headings() outputs: ")
            print(f"\nHeadings: {headings}, \nSub Headings: {subHeadings}, \nNot Required heading: {notRequired_Heading}, \nWork Project Headings: {Work_Project_Headings}\n\n")
            actualHeadingsCount = len(subHeadings)
            NRlength = len(notRequired_Heading)
            ORlength = len(Other_Headings)
            ORlength_db=len(Other_headings_db)


            actionwordsSet,actionwords_total,actionwords= get_action_words(text)
            print("🔍 get_action_words() outputs: ")
            print(f"\nAction Words Set: {actionwordsSet}, \nAction Words Total: {actionwords_total}, \nAction Words: {actionwords}\n\n")


            frequencyList,total_repeated_actionwords,repeated_frequency =  frequency_Action_words(actionwords)
            print("🔍 frequency_Action_words() outputs: ")
            print(f"\nFrequency List: {frequencyList}, \nTotal Repeated Actionwords: {total_repeated_actionwords}, \nRepeated Frequency: {repeated_frequency}\n\n")

            finalBullet,Bullets_Total,standard_bullet_flag= get_bullets(text)
            print("🔍 get_bullets() outputs: ")
            print(f"\nFinal Bullet: {finalBullet}, \nBullets Total: {Bullets_Total}, \nStandard Bullet: {standard_bullet_flag}\n\n")
            # bulletsType=len(finalBullet)
        
            
            ats_date =  getATS_dates(text)
            print("🔍 getATS_dates() outputs: ")
            print(f"\nATS Dates: {ats_date}\n\n")

            dates_nonAts =  get_nonATSdates(text)
            print("🔍 get_nonATSdates() outputs: ")
            print(f"\nNon ATS Dates: {dates_nonAts}\n\n")

            phonenumbers_finderSet,phonenumbers_finderSet1,phonenumbers_finderSet2,phone_all1 =  get_phones(text)
            print("🔍 get_phones() outputs: ")
            print(f"\nPhone Numbers: {phonenumbers_finderSet}, \nPhone Numbers1: {phonenumbers_finderSet1}, \nPhone Numbers2: {phonenumbers_finderSet2}, \nPhone All: {phone_all1}\n\n")

            measurable= get_namedEntityMeasurable(text)
            print("🔍 get_namedEntityMeasurable() outputs: ")
            print(f"\nMeasurable: {measurable}\n\n")
            # print("mesurable***************************************************",measurable)

            dates = ats_date + dates_nonAts

            clean_measurable =  get_measurableUpdated(text,finalBullet,measurable,phone_all1,dates)
            print("🔍 get_measurableUpdated() outputs: ")
            print(f"\nClean Measurable: {clean_measurable}\n\n")

            count_measurable = len(clean_measurable)
            

            total_word_count =  get_totalWordCount(text)
            print("🔍 get_totalWordCount() outputs: ")
            print(f"\nTotal Word Count: {total_word_count}\n\n")


            email_finderSet =  get_emails(text) 
            print("🔍 get_emails() outputs: ")
            print(f"\nEmails: {email_finderSet}\n\n")
            
            url,linkedIn_flag,url_flag,linkedIn =  get_url(pdf_file)
            print("🔍 get_url() outputs: ")
            print(f"\nURL: {url}, \nLinkedIn Flag: {linkedIn_flag}, \nURL Flag: {url_flag}, \nLinkedIn: {linkedIn}\n\n")
            linkedInUrl = linkedIn
            # # print(url)
            actionwordsSet_negative,actionwords_total_negative,actionwords1= get_negative_action_words(text)
            print("🔍 get_negative_action_words() outputs: ")
            print(f"\nNegative Action Words Set: {actionwordsSet_negative}, \nNegative Action Words Total: {actionwords_total_negative}, \nNegative Action Words: {actionwords1}\n\n")

            frequencyList_negative,total_repeated_actionwords_negative,repeated_frequency_negative =  frequency_Action_words(actionwords1)
            print("🔍 frequency_Action_words() outputs: ")
            print(f"\nFrequency List: {frequencyList_negative}, \nTotal Repeated Actionwords: {total_repeated_actionwords_negative}, \nRepeated Frequency: {repeated_frequency_negative}\n\n")
            

            file_size_kb,file_type,flag_file_size= get_fileDetails(pdf_file2)
            print("🔍 get_fileDetails() outputs: ")
            print(f"\nFile Size: {file_size_kb} kB, \nFile Type: {file_type}, \nFlag File Size: {flag_file_size}\n\n")
            # file_size_kb,file_type,flag_file_size= get_fileDetails(fake_pdf_file2)


            skills,Skills_Total =  extract_skills(text)
            print("🔍 extract_skills() outputs: ")
            print(f"\nSkills: {skills}, \nSkills Total: {Skills_Total}\n\n")


            images  =  check_Images(pdf_file)
            print("🔍 check_Images() outputs: ")
            print(f"\nImages: {images}\n\n")
            
            pages_count =  get_pageCount(pdf_file)
            print("🔍 get_pageCount() outputs: ")
            print(f"\nPages Count: {pages_count}\n\n")

            personalPronouns = get_excel_pronouns(text)
            print("🔍 get_excel_pronouns() outputs: ")
            print(f"\nPersonal Pronouns: {personalPronouns}\n\n")

            
            tables=get_tables(pdf_file)
            print("🔍 get_tables() outputs:")
            print(f"\nTables: {tables}\n\n")


            tables2=get_tables2(pdf_file)
            # print("tables2..........",tables2)
            print("🔍 get_tables2() outputs:")
            print(f"\nTables2: {tables2}\n\n")


            tables_flag=0
            if(tables=="table" and tables2 == "table"):
                tables_flag=1
            if(tables=="table"):
                tables_flag=1



            print("New Features")
            output_voice = text_voice(text)
            print("🔍 text_voice() outputs: ")
            print(f"\nVoice: {output_voice}\n\n")

            fillerwordsSet,fillerwords_total,fillerwords = get_filler_words(text)
            print("🔍 get_filler_words() outputs: ")
            print(f"\nFiller Words Set: {fillerwordsSet}, \nFiller Words Total: {fillerwords_total}, \nFiller Words: {fillerwords}\n\n")
            
            extra_words = ['\uf0d8','\uf0b7',':','/','|']
            combined_data = ats_date + dates_nonAts + phone_all1+finalBullet + list(email_finderSet) + extra_words
            combined_data = [word for sublist in combined_data for word in sublist.split()]
            raw_data2 = get_bold(pdf_file)
            print("🔍 get_bold() outputs: ")
            print(f"\nBold: {raw_data2}\n\n")
            # print("Bold --->",raw_data2)
            repeated_words = frequent_dynamic_ngrams(raw_data2,combined_data)
            print("🔍 frequent_dynamic_ngrams() outputs: ")
            print(f"\nRepeated Words: {repeated_words}\n\n")
            # print("Repeated done",repeated_words)

            total_Headings = headings + subHeadings
            output_Headings = [word for phrase in total_Headings for word in phrase.split()]
            print(output_Headings)

            reg_output = get_subWords(text)
            print("🔍 get_subWords() outputs:")
            print(f"\nSubwords: {reg_output}\n\n")

            reg_Numbers = get_Numbers(text)
            print("🔍 get_Numbers() outputs:")
            print(f"\nNumbers: {reg_Numbers}\n\n")

            reg_Phone = get_Phones(text)
            print("🔍 get_Phones() outputs:")
            print(f"\nPhones: {reg_Phone}\n\n")

            # print("hello")
            extra_words = [':','|','-','in','and','by','cbse','ssc','for','Phone','Phone:','Email:','Email','@',',','Page','Contact','Phone Number','Resume','CURRICULUM VITAE','CV','ID','Name','Location','&','*','?','of'] +  reg_Phone + finalBullet + skills + output_Headings+headings+subHeadings + list(email_finderSet) + url + phone_all1 + clean_measurable +reg_output + reg_Numbers
            # print("hello1")
            
            alloutput = detect_names_all(pdf_file,text,extra_words)
            print("🔍 detect_names_all() outputs: ")
            print(f"\nAll Output: {alloutput}\n\n")
            # print("hello2")
            # print(alloutput)

            name_output = get_finalise_names(alloutput)
            print("🔍 get_finalise_names() outputs: ")
            print(f"\nName Output: {name_output}\n\n")

            if match_strings(file_name, name_output):
                print("Strings match!")
            else:
                print("Strings do not match.")
            
            scored,score_array  = score(standard_font_style_flag,multiple_font_style,multiple_font_size,content_size_flag,font_colors_Total,standard_color_flag,actionwords_total,actionwords_total_negative,total_repeated_actionwords_negative,Bullets_Total,standard_bullet_flag,total_repeated_actionwords,ats_date,dates_nonAts,clean_measurable,total_word_count,email_finderSet,phonenumbers_finderSet,images,linkedIn_flag,pages_count,personalPronouns,tables_flag,Work_Project_Headings,EduSkill_Headings,NRlength,ORlength_db,Skills_Total,standard_match_headings_count,sectionMapCount,phone_all1,actualHeadingsCount,experience,fillerwords_total,output_voice,repeated_words,file_name, name_output)
            print("🔍 score() outputs: ")
            print(f"\nScored: {scored}, \nScore Array: {score_array}\n\n")

            
            comments,mark = get_chars(standard_font_style_flag,multiple_font_style,multiple_font_size,content_size_flag,font_colors_Total,standard_color_flag,actionwords_total,actionwords_total_negative,total_repeated_actionwords_negative,Bullets_Total,standard_bullet_flag,total_repeated_actionwords,
                                      ats_date,dates_nonAts,clean_measurable,total_word_count,email_finderSet,phonenumbers_finderSet,images,linkedIn_flag,
                                      url_flag,pages_count,personalPronouns,tables_flag,font_styles,font_sizes,max_size,actionwordsSet_negative,
                                      frequencyList_negative,phone_all1,skills,Work_Project_Headings,actionwordsSet,frequencyList,file_size_kb,headings,EduSkill_Headings,
                                      notRequired_Heading,Other_Headings,experience,file_name, name_output,output_voice,fillerwords)

            print("🔍 get_chars() outputs: ")
            print(f"\nComments: {comments}, \nMark: {mark}\n\n")



            comment_list, mark_list = get_commentList(repeated_words) 
            print("🔍 get_commentList() outputs: ")
            print(f"\nComments List: {comment_list}, \nMark List: {mark_list}\n\n")



            body={
                'font_styles': list(font_styles),
                'standard_font_style_flag':standard_font_style_flag,
                'multiple_font_style':multiple_font_style,
                'font_sizes':list(font_sizes),
                'multiple_font_size':multiple_font_size,
                'Content_Font_Size':max_size,
                'content_size_flag':content_size_flag,
                'font_colors':list(font_colors),
                'font_colors_Total':font_colors_Total,
                'standard_color_flag':standard_color_flag,
                'Sections_Found_Standard':list(headings),
                'Section_Headings_in_Resume':list(subHeadings),
                'actualHeadingsCount':actualHeadingsCount,
                'Work Experience_Projects':list(Work_Project_Headings),
                'EduSkill_Headings':list(EduSkill_Headings),
                'notRequired_Heading':list(notRequired_Heading),
                'NotRequiredHeadingsCount':NRlength,
                'Other Relevant Headings':list(Other_Headings),
                'OtherHeadingsCount':ORlength,
                'Other Relevant Section Headings':list(Other_headings_db),
                'Other Reveleant Section HeadingsCount':len(Other_headings_db),
                'Standard Match Headings':list(standard_match_headings),
                'Standard Match Headings   Count':standard_match_headings_count,
                
                'Section Count with Multiple Headings':sectionMapCount,
                'Action_Words':list(actionwordsSet),
                'actionwords_total':actionwords_total,
                'Repeated_Action_Words_Name':list(frequencyList),
                'total_repeated_actionwords':total_repeated_actionwords,
                'repeated_frequency':repeated_frequency,
                'finalBullet':list(finalBullet),
                'Bullets_total':Bullets_Total,
                'standard_bullet_flag':standard_bullet_flag,
                'ATS Dates':list(ats_date),
                'Non ATS Dates ':list(dates_nonAts),
                'Negative Action Words':list(actionwordsSet_negative),
                'Total Negative Action Words': actionwords_total_negative,
                'Repeated Negative Action Words Name':frequencyList_negative,
                'Total Repeated Negative Action Words':total_repeated_actionwords_negative,
                # 'Frequency Of Each Repeated Negative Action Words':my_string1,
                'measurable':list(clean_measurable),
                'Measurable Count':count_measurable,
                'url':url,
                'linkedin flag':linkedIn_flag,
                'Flag Url':url_flag,
                'total_word_count':total_word_count,
                'email_finderSet':list(email_finderSet),
                'phonenumbers':list(phone_all1),
                # # 'fileName':nameOfFile,
                'file size':file_size_kb,
                'file type':file_type,
                'skills':list(skills),
                'Skills_Total':Skills_Total,
                'images':images,
                'pages_count':pages_count,
                'personalPronouns':list(personalPronouns),
                'tables':tables_flag,
                'voice':output_voice,
                'fillerWords':list(fillerwords),
                'phrase':list(repeated_words.items()),
                'names':name_output,
                'Scores':scored,
                'Comments':comments,
                'Comment_mark': mark,
                'comment_list':comment_list,
                'mark_list': mark_list,
            }
            response = {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "OPTIONS,POST,GET"  # Adjust the allowed HTTP methods as needed
                },
                "body": json.dumps(body)
            }
            print(response)
        
        except Exception as e:
            print("lambda breaked")
            print(str(e))



        presentation=[]
        personalDetails=[]
        information=[]
        compentencies=[]
        personalDetails=[]
    
        presentation.append(comments[0])
        presentation.append(comments[1])
        presentation.append(comments[22])
        presentation.append(comments[18])
        presentation.append(comments[12])

        information.append(comments[5])
        information.append(comments[6])
        information.append(comments[7])
        information.append(comments[8])
        information.append(comments[3])
        information.append(comments[4])
        information.append(comments[19])
        information.append(comments[24])
        information.append(comments[23])


        compentencies.append(comments[9])
        compentencies.append(comments[10])
        compentencies.append(comments[11])
        compentencies.append(comments[16])
        compentencies.append(comments[20])
        compentencies.append(comments[21])

        personalDetails.append(comments[14])
        personalDetails.append(comments[13])
        personalDetails.append(comments[15])
        personalDetails.append(comments[17])



        presentation_mark=[]
        personalDetails_mark=[]
        information_mark=[]
        compentencies_mark=[]
    
        presentation_mark.append(mark[0])
        presentation_mark.append(mark[1])
        presentation_mark.append(mark[22])
        presentation_mark.append(mark[18])
        presentation_mark.append(mark[12])

        information_mark.append(mark[5])
        information_mark.append(mark[6])
        information_mark.append(mark[7])
        information_mark.append(mark[8])
        information_mark.append(mark[3])
        information_mark.append(mark[4])
        information_mark.append(mark[19])
        information_mark.append(mark[24])
        information_mark.append(mark[23])

        compentencies_mark.append(mark[9])
        compentencies_mark.append(mark[10])
        compentencies_mark.append(mark[11])
        compentencies_mark.append(mark[16])
        compentencies_mark.append(mark[20])
        compentencies_mark.append(mark[21])

        personalDetails_mark.append(mark[14])
        personalDetails_mark.append(mark[13])
        personalDetails_mark.append(mark[15])
        personalDetails_mark.append(mark[17])

        

        dta_highlight= {
            "measurableImpact": clean_measurable,
            "skills": skills,
            "fileSize": file_size_kb,
            "sections": subHeadings,
            "actionWords": actionwordsSet,
            "buzzWords": actionwordsSet_negative,
            "personalPronoun": personalPronouns,
            "fillerWords": fillerwords,
            "fontStyles": font_styles,
            "fontSizes": font_sizes,
            "wordCount": total_word_count,
            "phoneNumber": phone_all1,
            "emails": list(email_finderSet),
            "noRequiredHeading": notRequired_Heading,
            "additionalHeading": Other_Headings,
            "ats_dates" : list(ats_date),
            "non_ats_dates" : list(dates_nonAts),
            "activeVoice" : list(output_voice),
            "linkedIn" : linkedInUrl,
            "repeatedWords" : list(repeated_words) 
        }

        
        




        dta={
            "features":"true",
            "key":key,
            'score': scored,
            'personalDetails':personalDetails,
            'compentencies':compentencies,
            'information':information,
            'presentation':presentation,
            'presentation_mark': presentation_mark,
            'personalDetails_mark':personalDetails_mark,
            'information_mark':information_mark,
            'compentencies_mark':compentencies_mark,
            'comment_list':comment_list,
            'mark_list':mark_list,
            'dataHighlight': dta_highlight
        }



        print("rrrrrrrrrrrrrrrrrrrrrrrrrr22222222222222222222222222222222")
        print("key")
        print('data ', dta)
        # url = "https://fa54-2401-4900-5976-5d7b-1d2-10b1-fa62-c735.ngrok-free.app/features/lambdaResponse"
        headers = {"Content-Type": "application/json"}
        data = json.dumps(dta).encode('utf-8')
        try:
            # req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            print("sending requesttt")
            print("docker testingg................... requesttt")
            # response2 = requests.post(f"{root_url}/features/lambdaResponse", data=data, headers=headers, timeout=10)
            return response
            # response_server = urllib.request.urlopen(req)
            #  print("request sent to ec2 server.......................")
            # return response
            # response_data = response_server.read().decode("utf-8")
            # print(response_data)
            #  print("backkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
            
        except requests.exceptions.ReadTimeout:
            
            print("overrrrrrrrrrrr")
            response2 = None
            return response







        
       

    except Exception as e:
        print(e)
        print("erorrrrrrrrrrrr")
        body={
            'error': e,
        }
        response = {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET"  # Adjust the allowed HTTP methods as needed
            },
            "body": json.dumps(body)
        }