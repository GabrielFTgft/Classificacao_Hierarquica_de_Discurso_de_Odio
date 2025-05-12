import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import pandas as pd
import string

# Funcao para remover URLs e mencoes
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove menções
    
    stop_words = set(stopwords.words('portuguese'))
    my_stops_words = {'rt', 'RT', 'pra', 'vc', 'q', 'pq', 'http'}
    stop_words = stop_words.union(set(my_stops_words))

    return ' '.join([word for word in text.split() if word not in stop_words])

def preprocess_text(csv_file):
    df = pd.read_csv(csv_file)
    df['text'] = df['text'].apply(clean_text)
    print("Stop words removidas")
    return df