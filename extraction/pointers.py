from config.settings import *

def extract_first_words(bullets):
    first_words = []

    for bullet in bullets:
        if not bullet:
            continue

        # split by space and take first word
        first_word = bullet.strip().split()[0]

        first_words.append(first_word)

    return first_words

import pandas as pd

def load_action_words():
    df = pd.read_excel(ACTION_WORDS_EXCEL_FILE_PATH, usecols='A')
    words = df.iloc[:, 0].dropna().tolist()
    return set([w.lower() for w in words])  # set for O(1) lookup

def check_action_words(first_words):
    action_words_set = load_action_words()
    results = []

    for word in first_words:
        is_action = word.lower() in action_words_set

        results.append({
            "word": word,
            "is_action": is_action
        })

    return results

import spacy

nlp = spacy.load("en_core_web_sm")

def get_tense(word):
    doc = nlp(word)

    for token in doc:
        if token.pos_ == "VERB":
            tense = token.morph.get("Tense")
            
            if tense:
                return tense[0]  # Past / Pres
            else:
                # base form (like "Manage", "Lead")
                if token.tag_ == "VB":
                    return "Base"
    
    return "Unknown"

def analyze_first_words(first_words):
    action_words_set = load_action_words()
    results = []

    for word in first_words:
        is_action = word.lower() in action_words_set
        tense = get_tense(word) if is_action else None

        results.append({
            "word": word,
            "is_action": is_action,
            "tense": tense
        })

    return results

