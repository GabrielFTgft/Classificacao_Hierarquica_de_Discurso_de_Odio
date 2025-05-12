import pandas as pd
from paths import DATA_DIR

df = pd.read_csv(DATA_DIR / 'raw' /'portuguese_hate_speech_hierarchical_classification.csv')

print("Categories -- Total")
print(f"No Hate Speech -- {df.shape[0] - df[df.columns[1]].sum()}")
for i in range(1,80):
    nome = df.columns[i]
    soma = df[nome].sum()
    print(f"{nome} -- {soma}")


