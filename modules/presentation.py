from pdfminer.layout import LTTextContainer, LTChar, LTLine, LTAnno, LAParams
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
from pdfminer.high_level import extract_pages
from scipy.spatial import KDTree
from io import BytesIO

def get_maxSize_words(pdf_file):
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
