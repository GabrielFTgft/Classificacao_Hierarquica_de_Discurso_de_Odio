#usei o pickle para economizar tempo
import pickle
import numpy as np

embeddings_index = {}
with open('C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Dataset_Paula_Fortuna\\glove_s300.txt') as f:
    for line in f:
        value = line.split(' ')
        word = value[0]
        coefs = np.array(value[1:], dtype='float32')
        embeddings_index[word] = coefs

with open("embeddings_index.pkl", "wb") as f:
    pickle.dump(embeddings_index, f)
