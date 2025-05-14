import numpy as np
import pickle
from sklearn.metrics import classification_report
from scripts import pre_text_processing
from paths import DATA_DIR
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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

data = pre_text_processing.preprocess_text(DATA_DIR / 'preprocessed' / 'dataset_hierarchical_28.csv')

y = data.iloc[:, 1:]

with open(DATA_DIR / 'preprocessed' /'embeddings_index.pkl', "rb") as f:
    embeddings_index = pickle.load(f)

X = embed_text(data['text'], embeddings_index, 300)

print("Shape de X:", X.shape)

mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracyList = []
for train_index, test_index in mskf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    orc_model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'))

    orc_model.fit(X_train, y_train)

    y_pred = orc_model.predict(X_test)
    accuracy = orc_model.score(X_test, y_test)
    accuracyList.append(accuracy)
    print(classification_report(y_test, y_pred))

print("Media das acuracias:", np.mean(accuracyList))
