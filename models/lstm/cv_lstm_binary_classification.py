import pandas as pd
import numpy as np
import pickle
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from paths import DATA_DIR
from scripts import pre_text_processing

with open(DATA_DIR / 'preprocessed' /'embeddings_index.pkl', "rb") as f:
    embeddings_index = pickle.load(f)

tweets = pre_text_processing.preprocess_text(DATA_DIR / 'preprocessed' / 'hate_speech_binary_dataset.csv')

max_len = 100  #comprimento máximo da sequência - alterar depois
tokenizer = Tokenizer() # Inicializa o tokenizer
tokenizer.fit_on_texts(tweets['text'])  # Ajusta o tokenizer aos seus tokens

word_index = tokenizer.word_index
tam_voc = len(word_index) + 1

# Padronizar as sequências
sequences = tokenizer.texts_to_sequences(tweets['text'])  # Converter os textos para índices
data = pad_sequences(sequences, maxlen=max_len)  # Ajustar o comprimento das sequências

labels = tweets['class']

embedding_matrix = np.zeros((tam_voc, 300)) #300 é a dimensão do Glove baixado

#Para cada palavra e seu índice no dicionário, tenta encontrar o vetor de embedding
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word.strip("'"))
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

def create_model(embedding_matrix, max_len, tam_voc):

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

    output_layer = Dense(1, activation='sigmoid')(drop2)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compilando o modelo
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

scores = []

k = 10
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=88)
fold = 1.

for train_index, test_index in kf.split(data, labels):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # # Converter as sequências de volta para texto
    # X_train_text = tokenizer.sequences_to_texts(X_train)  # Converter para texto
    # X_test_text = tokenizer.sequences_to_texts(X_test)    # Converter para texto

    # # Salvar os arquivos CSV
    # train_file = f"C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Codigos\\Binaria\\CV_files\\train_fold{fold}.csv"
    # test_file = f"C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Codigos\\Binaria\\CV_files\\test_fold{fold}.csv"
    # pd.DataFrame({'text': X_train_text, 'class': y_train}).to_csv(train_file, index=False)
    
    model = create_model(embedding_matrix, max_len, tam_voc)
    # Treinando o modelo
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Avaliação do modelo
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy do fold {fold}: {score[1] * 100:.2f}%")
    scores.append(score[1])  # Armazenar apenas a precisão sem multiplicar por 100

    # Relatório de classificação
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    # pd.DataFrame({'text': X_test_text, 'class': y_test, 'pred': y_pred.flatten()}).to_csv(test_file, index=False)

    print(f"Relatório de classificação do fold {fold}:")
    print(classification_report(y_test, y_pred))
    fold += 1

# Calcular a média e o desvio padrão dos scores
media = np.mean(scores)
desvio_padrao = np.std(scores)

print(f"CV score: {media * 100:.2f}%")
print(f"CV desvio padrão: {desvio_padrao:.2f}%")
