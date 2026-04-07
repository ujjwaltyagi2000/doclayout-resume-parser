import os
import nltk

DATA_DIR = os.path.join(os.getcwd(), "data")

MODEL_PATH = "model/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
SECTIONS_OUPUT_FILE_PATH = "sections_output.json"
STANDARD_SECTIONS_OUTPUT_FILE_PATH = "standard_sections_output.json"
OUTPUT_FILE_PATH = "analysis_output.json"
nltk.data.path.append("nltk_data")

# Excel Sheets
ACTION_WORDS_EXCEL_FILE_NAME = "ActionWords.xlsx"
NEGATIVE_ACTION_WORDS_EXCEL_FILE_NAME = "Negative Action Words.xlsx"
FILLER_WORDS_EXCEL_FILE_NAME = "Filler Words.xlsx"

ACTION_WORDS_EXCEL_FILE_PATH = os.path.join(DATA_DIR, ACTION_WORDS_EXCEL_FILE_NAME)
NEGATIVE_ACTION_WORDS_EXCEL_FILE_PATH = os.path.join(DATA_DIR, NEGATIVE_ACTION_WORDS_EXCEL_FILE_NAME)
FILLER_WORDS_EXCEL_FILE_PATH = os.path.join(DATA_DIR, FILLER_WORDS_EXCEL_FILE_NAME)