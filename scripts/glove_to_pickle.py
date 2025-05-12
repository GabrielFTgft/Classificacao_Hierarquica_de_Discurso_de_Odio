import pickle
import numpy as np
from pathlib import Path
from paths import DATA_DIR

index_path = Path(__file__).resolve().parent.parent / 'data' / 'preprocessed' / 'embeddings_index.pkl'


embeddings_index = {}
with open(DATA_DIR / 'raw' /'glove_s300.txt') as f:
    for line in f:
        value = line.split(' ')
        word = value[0]
        coefs = np.array(value[1:], dtype='float32')
        embeddings_index[word] = coefs

with open(index_path, "wb") as f:
    pickle.dump(embeddings_index, f)
