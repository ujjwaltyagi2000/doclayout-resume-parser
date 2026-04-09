from config.settings import *
import pandas as pd
import nltk

def get_excel_pronouns(text):
    try:
        dataframe1 = pd.read_excel(PRONOUNS_EXCEL_FILE_PATH,usecols='A')
        
        SKILLS_DB = dataframe1.values.tolist()
        topics_flat1 = [topic for sublist in SKILLS_DB for topic in sublist]
        topics_flat =  [x.lower() for x in topics_flat1]
        print(topics_flat)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = nltk.tokenize.word_tokenize(text)
        
        
        filtered_tokens = [w for w in word_tokens if w not in stop_words]
    
        
        filtered_tokens = [w for w in word_tokens if w.isalpha()]
    

        bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))   
    

        pronoun = []

        for token in filtered_tokens:
            
            if token.lower() in topics_flat:
            
                pronoun.append(token.lower())

        for ngram in bigrams_trigrams:
            if ngram.lower() in topics_flat:
                pronoun.append(ngram.lower())
        
        return list(pronoun)
    except Exception as e:
        print("get_pronoun_words ",str(e)) 
        return []