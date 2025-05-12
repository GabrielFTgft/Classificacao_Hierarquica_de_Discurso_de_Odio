import pandas as pd
import numpy as np
import pickle
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from paths import DATA_DIR
from scripts import pre_text_processing

def create_array(df):
    ndf = pd.DataFrame(columns=['text', 'class'])

    lista_classes= []
    ndf['text'] = df['text']
    del df['text']
    for index, row in df.iterrows():
        classes = [r for r in row] 
        lista_classes.append(classes)

    ndf['class'] = lista_classes
    return ndf


with open(DATA_DIR / 'preprocessed' /'embeddings_index.pkl', "rb") as f:
    embeddings_index = pickle.load(f)

tweets = pre_text_processing.preprocess_text(DATA_DIR / 'preprocessed' / 'dataset_hierarchical_28.csv')

max_len = 100  #comprimento máximo da sequência - alterar depois
tokenizer = Tokenizer() # Inicializa o tokenizer
tokenizer.fit_on_texts(tweets['text'])  # Ajusta o tokenizer aos seus tokens

sequences = tokenizer.texts_to_sequences(tweets['text'])  #Converte os tokens em sequências de índices

word_index = tokenizer.word_index
tam_voc = len(word_index) + 1

# Padronizar as sequências
sequences = tokenizer.texts_to_sequences(tweets['text'])  # Converter os textos para índices
data = pad_sequences(sequences, maxlen=max_len)  # Ajustar o comprimento das sequências

embedding_matrix = np.zeros((tam_voc, 300)) #300 é a dimensão do Glove baixado

#Para cada palavra e seu índice no dicionário, tenta encontrar o vetor de embedding
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

tweets = create_array(tweets)

# cada twwet['class'] conte uma lista representadno as 79 classes [0,1,0,1,1,0,1,0,;;;;;;;;;;;;;]
labels = np.array(tweets['class'].tolist(), dtype=np.int32)
numClasses = 28

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

input_layer = Input(shape=(max_len,))

# Camada de embedding
embedding = Embedding(
    input_dim=tam_voc,          # Tamanho do vocabulário
    output_dim=300,             # Dimensão do embedding (GloVe = 300)
    weights=[embedding_matrix], # Pesos da matriz de embeddings pré-treinada
    input_length=max_len,       # Comprimento máximo da sequência
    trainable=False             # Embeddings fixos
)(input_layer)

drop1 = Dropout(0.5)(embedding)

# Camada LSTM
lstm = LSTM(50, return_sequences=False)(drop1)

drop2 = Dropout(0.5)(lstm)

output_layer = Dense(numClasses, activation='sigmoid')(drop2)

model = Model(inputs=input_layer, outputs=output_layer)
# Compilando o modelo
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

# Treinando o modelo
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1
)

#Avaliação do modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {scores[1] * 100:.2f}%")

# Relatório de classificação
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(y_pred)
print(classification_report(y_test, y_pred))

