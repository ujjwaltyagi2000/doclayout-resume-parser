from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTAnno, LAParams
from io import BytesIO
from pdfminer.high_level import extract_pages

def get_bold(pdf_file):
    
    try: 

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