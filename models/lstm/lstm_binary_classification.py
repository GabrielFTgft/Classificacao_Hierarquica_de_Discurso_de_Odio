import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from paths import DATA_DIR
from scripts import pre_text_processing

with open(DATA_DIR / 'preprocessed' /'embeddings_index.pkl', "rb") as f:
    embeddings_index = pickle.load(f)

tweets = pre_text_processing.preprocess_text(DATA_DIR / 'preprocessed' / 'hate_speech_binary_dataset.csv')

max_len = 100  #comprimento máximo da sequência - alterar depois
tokenizer = Tokenizer() # Inicializa o tokenizer
tokenizer.fit_on_texts(tweets['text'])  # Ajusta o tokenizer aos seus tokens

sequences = tokenizer.texts_to_sequences(tweets['text'])  #Converte os tokens em sequências de índices

word_index = tokenizer.word_index

tam_voc = len(word_index) + 1

embedding_matrix = np.zeros((tam_voc, 300)) #300 é a dimensão do Glove baixado

#Para cada palavra e seu índice no dicionário, tenta encontrar o vetor de embedding
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word.strip("'"))
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Padronizando o comprimento das sequências
data = pad_sequences(sequences, maxlen=max_len)

# Separando as classes
labels = tweets['class']  # Assumindo que a coluna de classes é chamada 'classes'

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42, stratify=labels)

model = Sequential()
model.add(Embedding(input_dim=tam_voc, output_dim=300, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(Dropout(0.5))  # ← Após o embedding
model.add(LSTM(50))
model.add(Dropout(0.5))  # ← Após a LSTM
model.add(Dense(1, activation='sigmoid'))

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
    validation_split=0.1,
    verbose=1
)

# Relatório de classificação
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

feature_extractor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# Extrair características da LSTM
X_train_lstm = feature_extractor.predict(X_train)
X_test_lstm = feature_extractor.predict(X_test)

# Definir o classificador XGBoost base
xgb_model = xgb.XGBClassifier(
    #objective='binary:logistic', !!!
    eval_metric='logloss',
)

# Definir o grid de hiperparâmetros
param_grid = {
    'eta': [0.0, 0.3, 1.0],  # Taxa de aprendizado
    'gamma': [0.1, 1, 10]    # Controle de crescimento da árvore
}

# Criar o GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=10, 
    verbose=2, #!!! 
    n_jobs=-1 #!!!
)

grid_search.fit(X_train_lstm, y_train)

print("Best eta: ",grid_search.best_params_['eta'])
print("Best gamma: ",grid_search.best_params_['gamma'])
print(f"Grid best score: {grid_search.best_score_ * 100:.2f}%")

# best_xgb = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',
#     eta=grid_search.best_params_['eta'],
#     gamma=grid_search.best_params_['gamma']
# )

# best_xgb.fit(X_train_lstm, y_train)

best_xgb = grid_search.best_estimator_

y_pred_xgb = best_xgb.predict(X_test_lstm)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {acc_xgb * 100:.2f}%")
# Relatório de classificação
print(classification_report(y_test, y_pred_xgb))
