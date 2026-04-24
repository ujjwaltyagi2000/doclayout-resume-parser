"""

Script to filter out body content from a PDF using font size as a heuristic. It identifies the most common font size as the body text and extracts lines of text that do not match this font size, which are likely to be headings or other non-body content.

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
    
# Same function as above, but returns word_positions as well for Yolo mapping
# 🧪 Currently in testing
def filter_body_content_new(pdf_file):
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
        
        return result, font_and_words, max_size, len(max_words), word_positions

    except Exception as e:
        print(f"❌ Error in filter_body_content: {e}")
        return []