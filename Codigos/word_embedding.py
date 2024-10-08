import numpy as np
import pandas as pd
import pickle
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
#from keras import layers

with open("C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Codigos\\embeddings_index.pkl", "rb") as f:
    embeddings_index = pickle.load(f)

tweets = pd.read_csv('C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Arquivos_Gerados\\dataset_noStopWords.csv')


max_len = 25  #comprimento máximo da sequência - alterar depois
tokenizer = Tokenizer() # Inicializa o tokenizer
tokenizer.fit_on_texts(tweets['tokens'])  # Ajusta o tokenizer aos seus tokens

sequences = tokenizer.texts_to_sequences(tweets['tokens'])  #Converte os tokens em sequências de índices

word_index = tokenizer.word_index
#print(word_index)

# Padronizando o comprimento das sequências
#data = pad_sequences(sequences, maxlen=max_len)
tam_voc = len(word_index) + 1

hits = 0
misses = 0
embedding_matrix = np.zeros((tam_voc, 300)) #300 é a dimensão do Glove baixado

#Para cada palavra e seu índice no dicionário, tenta encontrar o vetor de embedding
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word.strip("'"))
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

print(embedding_matrix)
print("Converted %d words (%d misses)" % (hits, misses))

