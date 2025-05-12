import pandas as pd
from paths import DATA_DIR

df = pd.read_csv(DATA_DIR / 'raw' /'portuguese_hate_speech_hierarchical_classification.csv')

file_path = DATA_DIR / 'preprocessed' / 'hate_speech_binary_dataset.csv'
ndf = pd.DataFrame(columns=['text', 'class'])

ndf['text'] = df['text']
ndf['class'] = df['Hate.speech']

ndf.to_csv(file_path, index=False)
print(ndf.head())
print("Classificacao binaria concluida")