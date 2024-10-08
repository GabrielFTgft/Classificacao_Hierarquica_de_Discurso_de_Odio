import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string

tweets = pd.read_csv('C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Arquivos_Gerados\\hate_speech_classification.csv')

#conversão para letras minúsculas
tweets['text']  =  tweets['text'].str.lower()
#remoção de pontuação
tweets['text'] = tweets['text'].str.replace(f"[{string.punctuation}]", "", regex=True)

tokens = pd.DataFrame({
    'tokens': tweets['text'].apply(word_tokenize),
    'class': tweets['class']
})

print("Tokenizacao concluida")

tokens.to_csv('dataset_tokenizados.csv', index=True)