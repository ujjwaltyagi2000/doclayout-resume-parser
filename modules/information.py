import pandas as pd
import nltk
from collections import Counter
from spacy.matcher import Matcher
import en_core_web_sm
nlp = en_core_web_sm.load()
from config.settings import *

# ACTION WORDS EXTRACTION
def get_action_words(text):
    try:
        print("✅ Inside get_action_words function")
        # ACTION_WORDS_EXCEL_FILE_PATH = 'ActionWords.xlsx'
        dataframe1 = pd.read_excel(ACTION_WORDS_EXCEL_FILE_PATH,usecols='A')
        
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
    

# NEGATIVE ACTION WORDS/BUZZWORDS EXTRACTION

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
        
        # dataframe2 = pd.read_excel('Negative Action Words.xlsx',usecols='A')
        dataframe2 = pd.read_excel(NEGATIVE_ACTION_WORDS_EXCEL_FILE_PATH,usecols='A')
        

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
    
def get_filler_words(text):
    try:
        print("✅ Inside get_filler_words function")
        # file_path = 'Filler Words.xlsx'
        dataframe1 = pd.read_excel(FILLER_WORDS_EXCEL_FILE_PATH, usecols='A')
        
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
    
if __name__ == "__main__":
    
    text = "Led a team of 5 software engineers to develop a new feature that increased user engagement by 20%."
    action_words, total_action_words, all_action_words = get_action_words(text)
    print("Action Words:", action_words)
    print("Total Action Words:", total_action_words)
    print("All Action Words:", all_action_words)

    frequency_list, total_frequent_action_words, repeated_frequency = frequency_Action_words(all_action_words)
    print("Frequent Action Words (appearing at least 3 times):", frequency_list)
    print("Total Frequent Action Words:", total_frequent_action_words)
    print("Repeated Frequency of Action Words:", repeated_frequency)

    negative_action_words, total_negative_action_words, all_negative_action_words = get_negative_action_words(text)
    print("Negative Action Words:", negative_action_words)
    print("Total Negative Action Words:", total_negative_action_words)
    print("All Negative Action Words:", all_negative_action_words)

    filler_words, total_filler_words, all_filler_words = get_filler_words(text)
    print("Filler Words:", filler_words)
    print("Total Filler Words:", total_filler_words)
    print("All Filler Words:", all_filler_words)

    voice = text_voice(text)
    print("Passive Voice Constructions:", voice)