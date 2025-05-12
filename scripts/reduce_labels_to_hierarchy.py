import pandas as pd
from pathlib import Path
from paths import DATA_DIR

df = pd.read_csv(DATA_DIR / 'raw' /'portuguese_hate_speech_hierarchical_classification.csv')

minimo = 10 
agrupar = True

while agrupar:
    agrupar = False
    for col_nome in df.columns:
        if col_nome != 'text':  
            soma = df[col_nome].sum()
            if soma < minimo:  
                df.drop(columns=[col_nome], inplace=True) 
                agrupar = True 

file_path = Path(__file__).resolve().parent.parent / 'data' / 'preprocessed' / 'dataset_hierarchical_28.csv'
df.to_csv(file_path, index=False)

print("Categories -- Total")
print(f"No Hate Speech -- {df.shape[0] - df['Hate.speech'].sum()}")
for col_nome in df.columns:
    if col_nome != 'text':  
        soma = df[col_nome].sum()
        print(f"{col_nome} -- {soma}")



