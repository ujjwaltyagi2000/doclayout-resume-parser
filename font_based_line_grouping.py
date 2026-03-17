
"""
Built over resuscan_getheadings.py.

Also returns the font sizes of the headings
"""


import pathlib
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    TYPE_CHECKING,
    cast,
)
import io
from io import BytesIO
from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTAnno, LAParams
import pandas as pd
import pdfplumber
from pdfminer.pdfpage import PDFPage
import nltk

from typing import Any, BinaryIO, Container, Iterator, Optional, cast
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTAnno, LAParams
import re


FileOrName = Union[pathlib.PurePath, str, io.IOBase]

def extract_headings_two(words_to_check_flat):
            try:
                # Read the Excel file into a pandas DataFrame
                
                
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

def matchCategories(word_list):
            try: 
                # file_path ='Headlines.xlsx'
                file_path = 'https://s3.ap-south-1.amazonaws.com/mployee.me/keywords_list/Headlines.xlsx'
            
                df = pd.read_excel(file_path)

                # Create empty dictionary
                word_dict = {}

                # Loop through each row in the dataframe
                for index, row in df.iterrows():
                    
                    # Get key and value from row
                    # key = row[0]
                    # value = row[1]

                    key = row.iloc[0]
                    value = row.iloc[1]


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

def standard_headingsMatch(word_list):
            try: 
                
            

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

def filter_body_content(pdf_file):
    try: 
        word_to_size = {}
        size_to_word = {}
        word_positions = []  # Store (word, font_size, y_coordinate)

        for page_layout in extract_pages(BytesIO(pdf_file)):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_object in element:
                        if isinstance(text_object, (LTChar, LTAnno)):
                            continue
                        
                        font_size = None
                        current_word = ''
                        y_coord = None
                        
                        for character in text_object:
                            if isinstance(character, LTChar):
                                if font_size is None:
                                    font_size = round(character.size)
                                if y_coord is None:
                                    y_coord = round(character.y0, 1)  # Get y-coordinate
                                
                                if character.get_text().isspace():
                                    if current_word:
                                        # Store word with position
                                        word_positions.append((current_word, font_size, y_coord))
                                        
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
                                    y_coord = None
                                else:
                                    current_word += character.get_text()
                                    font_size = round(character.size)
                                    y_coord = round(character.y0, 1)
                        
                        if current_word:
                            word_positions.append((current_word, font_size, y_coord))
                            if current_word not in word_to_size:
                                word_to_size[current_word] = [font_size]
                            else:
                                word_to_size[current_word].append(font_size)
                            if font_size not in size_to_word:
                                size_to_word[font_size] = [current_word]
                            else:
                                size_to_word[font_size].append(current_word)

        # Find the body font size (most common)
        max_size = None
        max_words = []
        font_and_words = []
        for size, words in size_to_word.items():
            font_and_words.append({"size":size, "words": len(words)})
            print(f"🚀 Size: {size}, Words: {len(words)}")
            if max_size is None or len(words) > len(max_words):
                max_size = size
                max_words = words
        
        print(f"✅ Body Font Size: {max_size}, Words: {len(max_words)}")

        # Filter out body content and group by lines
        lines_dict = {}  # {(y_coord, font_size): [words]}
        
        for word, font_size, y_coord in word_positions:
            if font_size != max_size:  # Exclude body content
                key = (y_coord, font_size)
                if key not in lines_dict:
                    lines_dict[key] = []
                lines_dict[key].append(word)
        
        # Sort by y-coordinate (descending, top to bottom)
        sorted_lines = sorted(lines_dict.items(), key=lambda x: x[0][0], reverse=True)
        
        # Format output as lines
        result = []
        for (y_coord, font_size), words in sorted_lines:
            line_text = ' '.join(words)
            result.append({
                'text': line_text,
                'font': font_size,
                'y_position': y_coord
            })
        
        return result, font_and_words, max_size, len(max_words)

    except Exception as e:
        print(f"❌ Error in filter_body_content: {e}")
        return []

def get_full_content_with_fonts(pdf_file):
    """Returns ALL lines (body + headers) with font size and y_position"""
    try:
        word_positions = []
        size_to_word = {}

        for page_layout in extract_pages(BytesIO(pdf_file)):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_object in element:
                        if isinstance(text_object, (LTChar, LTAnno)):
                            continue
                        
                        font_size = None
                        current_word = ''
                        y_coord = None
                        
                        for character in text_object:
                            if isinstance(character, LTChar):
                                if font_size is None:
                                    font_size = round(character.size)
                                if y_coord is None:
                                    y_coord = round(character.y0, 1)
                                
                                if character.get_text().isspace():
                                    if current_word:
                                        word_positions.append((current_word, font_size, y_coord))
                                        size_to_word.setdefault(font_size, []).append(current_word)
                                    current_word = ''
                                    font_size = None
                                    y_coord = None
                                else:
                                    current_word += character.get_text()
                                    font_size = round(character.size)
                                    y_coord = round(character.y0, 1)
                        
                        if current_word:
                            word_positions.append((current_word, font_size, y_coord))
                            size_to_word.setdefault(font_size, []).append(current_word)

        # Group into lines by (y_coord, font_size)
        lines_dict = {}
        for word, font_size, y_coord in word_positions:
            key = (y_coord, font_size)
            lines_dict.setdefault(key, []).append(word)

        # Sort top to bottom (descending y)
        sorted_lines = sorted(lines_dict.items(), key=lambda x: x[0][0], reverse=True)

        full_content = []
        for (y_coord, font_size), words in sorted_lines:
            full_content.append({
                'text': ' '.join(words),
                'font': font_size,
                'y_position': y_coord
            })

        return full_content

    except Exception as e:
        print(f"❌ Error in get_full_content_with_fonts: {e}")
        return []

if __name__ == "__main__":

    # pdf_file_path = "Puunita Chaturvedi.pdf"
    # pdf_file_path = "resume.pdf"
    # pdf_file_path = "Ujjwal Tyagi.pdf"
    pdf_file_path = "TANVI GAWALI CV.pdf"
    # pdf_file_path = "Megha resume.pdf"
    
    # Read the PDF file as bytes
    with open(pdf_file_path, 'rb') as f:
        pdf_file = f.read()

    lines = filter_body_content(pdf_file)
    
    # Print line by line
    # print(lines)

    
    filtered_data = [
        {k: v for k, v in item.items() if k != 'y_position'}
        for item in lines
    ]

    print(filtered_data)