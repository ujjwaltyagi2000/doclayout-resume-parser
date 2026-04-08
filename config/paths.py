import os
import nltk

MODEL_PATH = "model/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
nltk.data.path.append("nltk_data")

DATA_DIR = os.path.join(os.getcwd(), "data")
JSON_DIR = os.path.join(os.getcwd(), "json")

for dir_path in [DATA_DIR, JSON_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


SECTIONS_OUTPUT_FILE_NAME = "sections_output.json"
STANDARD_SECTIONS_OUTPUT_FILE_NAME = "standard_sections_output.json"
FINAL_OUTPUT_FILE_NAME = "analysis_output.json"

SECTIONS_OUTPUT_FILE_PATH = os.path.join(JSON_DIR, SECTIONS_OUTPUT_FILE_NAME)
STANDARD_SECTIONS_OUTPUT_FILE_PATH = os.path.join(JSON_DIR, STANDARD_SECTIONS_OUTPUT_FILE_NAME)
FINAL_OUTPUT_FILE_PATH = os.path.join(JSON_DIR, FINAL_OUTPUT_FILE_NAME)

# Excel Sheets
ACTION_WORDS_EXCEL_FILE_NAME = "ActionWords.xlsx"
NEGATIVE_ACTION_WORDS_EXCEL_FILE_NAME = "Negative Action Words.xlsx"
FILLER_WORDS_EXCEL_FILE_NAME = "Filler Words.xlsx"
MEASURABLES_EXCEL_FILE_NAME = "Measurable.xlsx"

# Excel Sheets on S3
KEYWORDS_EXCEL_FILE_PATH = "https://s3.ap-south-1.amazonaws.com/mployee.me/keywords_list/Keywords.xlsx"

ACTION_WORDS_EXCEL_FILE_PATH = os.path.join(DATA_DIR, ACTION_WORDS_EXCEL_FILE_NAME)
NEGATIVE_ACTION_WORDS_EXCEL_FILE_PATH = os.path.join(DATA_DIR, NEGATIVE_ACTION_WORDS_EXCEL_FILE_NAME)
FILLER_WORDS_EXCEL_FILE_PATH = os.path.join(DATA_DIR, FILLER_WORDS_EXCEL_FILE_NAME)
MEASURABLES_EXCEL_FILE_PATH = os.path.join(DATA_DIR, MEASURABLES_EXCEL_FILE_NAME)