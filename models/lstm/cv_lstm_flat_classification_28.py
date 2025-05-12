from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import pickle
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from paths import DATA_DIR
from scripts import pre_text_processing
from sklearn.metrics import accuracy_score

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

def create_model(embedding_matrix, max_len, tam_voc, num_classes):
    input_layer = Input(shape=(max_len,))

    embedding = Embedding(
        input_dim=tam_voc,
        output_dim=300,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False
    )(input_layer)

    drop1 = Dropout(0.5)(embedding)
    lstm = LSTM(50, return_sequences=False)(drop1)
    drop2 = Dropout(0.5)(lstm)
    output_layer = Dense(num_classes, activation='sigmoid')(drop2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


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
labels = np.array(tweets['class'].tolist(), dtype=np.int32)
numClasses = 28

# Validação cruzada
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=88)
acc_folds = []
fold = 1

for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    model = create_model(embedding_matrix, max_len, len(tokenizer.word_index) + 1, numClasses)
    
    print(f"Treinando o modelo para o fold {fold}...")

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(f"Relatório de classificação do fold {fold}:")
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    acc_folds.append(acc)
    fold += 1

# Results
acc_mean = np.mean(acc_folds)

print(f"Accuracy mean: {acc_mean * 100:.2f}%")
