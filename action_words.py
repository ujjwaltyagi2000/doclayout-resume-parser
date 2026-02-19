import nltk
import fitz
import pandas as pd

from io import BytesIO

nltk.data.path.append("nltk_data")

def pdfText(pdf_bytes):
    """
    Extracts text from a PDF using PyMuPDF (fitz).

    Args:
        pdf_bytes (bytes): The PDF file content as bytes.

    Returns:
        str: The extracted text from the PDF.
    """
    try:
        print("✅ Inside pdfText function (using fitz)")

        # Open PDF directly from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""
        for i, page in enumerate(doc):
            print(f"🔹 Extracting text from page {i+1}")
            text += page.get_text("text")

        doc.close()
        print("✅ Text extraction complete")
        return text.strip()

    except Exception as e:
        print(f"❌ Error in pdfText: {e}")
        return ""


def get_action_words(text):
    try:
        print("✅ Inside get_action_words function")
        file_path = 'ActionWords.xlsx'
        dataframe1 = pd.read_excel(file_path,usecols='A')
        
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
        return list(actionwordsSet),actionwords_total,list(actionwords), len(actionwordsSet)
    except Exception as e:
        print("get_action_words ",e) 
        return [],0,[], 0
    
if __name__ == "__main__":

    key = "Ujjwal Tyagi.pdf"
    pdf_file_path = key
    
    # Read the PDF file as bytes
    with open(pdf_file_path, 'rb') as f:
        pdf_file = f.read()
    # pdf_file = key
    # text = pdfText(key)
    text = pdfText(pdf_file)

    actionwordsSet,actionwords_total,actionwords, len_action_words_set= get_action_words(text)
    print("🔍 get_action_words() outputs: ")
    print(f"\nAction Words Set: {actionwordsSet}, \nAction Words Set Length: {len_action_words_set},\nAction Words: {actionwords}, \nAction Words Total: {actionwords_total} \n\n")
