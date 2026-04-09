from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTAnno, LAParams
from io import BytesIO
from pdfminer.high_level import extract_pages

from modules.competencies import *

def get_finalise_names(alloutput):
    if alloutput is None:
        return ""
    elif len(alloutput) < 3:
        return ""
    else:
        return alloutput    

def get_max_size(pdf_file,txt):
   
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
    
def detect_names(text):
    try:
        doc = nlp(text)
        names = []
        
        for entity in doc.ents:
            
            if entity.label_ == "PERSON":
            
                names.append(entity.text)
        return names
    except Exception as e:
        print("******detect_names*****")
        print(str(e))
        
def detect_names_all(pdf_file,raw_data,extra_words):
    try:
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