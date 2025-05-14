import numpy as np
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from paths import DATA_DIR
from scripts import pre_text_processing
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

def embed_text(texts, embeddings_index, dim):
    embed = []
    for text in texts:
        words = text.split()
        vectors = [embeddings_index[word] for word in words if word in embeddings_index]
        if vectors:
            mean_vec = np.mean(vectors, axis=0)
        else:
            mean_vec = np.zeros(dim)
        embed.append(np.array(mean_vec).flatten()) 
    return np.array(embed)   

data = pre_text_processing.preprocess_text(DATA_DIR / 'preprocessed' / 'hate_speech_binary_dataset.csv')

y = data['class']

with open(DATA_DIR / 'preprocessed' /'embeddings_index.pkl', "rb") as f:
    embeddings_index = pickle.load(f)

X = embed_text(data['text'], embeddings_index, 300)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) #eh possivel adicionar a estraticacao com o parametro stratify

orc_model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'))

orc_model.fit(X_train, y_train)

y_pred = orc_model.predict(X_test)
print("Acuracia:", orc_model.score(X_test, y_test))

print(classification_report(y_test, y_pred))

