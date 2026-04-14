from config.settings import *
import pandas as pd
import spacy

# 🔥 Load spaCy once
nlp = spacy.load("en_core_web_sm")

# 🔥 Load action words once (lemma-based)
def load_action_words_lemma():
    df = pd.read_excel(ACTION_WORDS_EXCEL_FILE_PATH, usecols='A')
    words = df.iloc[:, 0].dropna().tolist()

    lemma_set = set()

    for word in words:
        doc = nlp(word)
        for token in doc:
            lemma_set.add(token.lemma_.lower())

    return lemma_set

ACTION_LEMMA_SET = load_action_words_lemma()

pd.DataFrame(list(ACTION_LEMMA_SET), columns=["Action Lemma"]).to_csv("action_lemmas.csv", index=False)

# Extract first words
def extract_first_words(bullets):
    first_words = []

    for bullet in bullets:
        if not bullet:
            continue

        words = bullet.strip().split()
        if not words:
            continue

        first_words.append(words[0])

    return first_words


# Tense Detection
def get_tense(token):
    tense = token.morph.get("Tense")

    if tense:
        return tense[0]  # Past / Pres

    if token.tag_ == "VBG":
        return "Present-Continuous"
    elif token.tag_ == "VB":
        return "Base"

    return "Unknown"


# Suggest Past Tense
def suggest_past_tense(token):
    lemma = token.lemma_

    if lemma.endswith("e"):
        return lemma + "d"
    else:
        return lemma + "ed"


# Main Analyzer
def analyze_first_words(first_words):
    results = []

    for word in first_words:
        doc = nlp(word)

        for token in doc:
            lemma = token.lemma_.lower()
            is_action = lemma in ACTION_LEMMA_SET

            tense = None
            suggestion = None

            if is_action:
                tense = get_tense(token)

                if tense != "Past":
                    suggestion = suggest_past_tense(token)

            results.append({
                "original": word,
                "lemma": lemma,
                "is_action": is_action,
                "tense": tense,
                "needs_fix": is_action and tense != "Past",
                "suggested": suggestion
            })

    return results