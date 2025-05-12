import numpy as np
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout, Input

def treinar_classificador_multiclasse(X_train, X_test, y_train, y_test, tam_voc, num_classes, word_index, embeddings_index):
    embedding_matrix = np.zeros((tam_voc, 300)) #300 é a dimensão do Glove baixado

    #Para cada palavra e seu índice no dicionário, tenta encontrar o vetor de embedding
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    max_len = 100

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

    output_layer = Dense(num_classes, activation='sigmoid')(drop2)

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
        validation_split=0.1,
        verbose=1
    )

    #Avaliação do modelo
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy: {score[1] * 100:.2f}%")
    
    return model, score[1] * 100