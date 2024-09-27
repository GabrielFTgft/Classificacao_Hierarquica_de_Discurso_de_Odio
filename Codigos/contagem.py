import pandas as pd
df = pd.read_csv('C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\dataset_Paula_Fortuna\\portuguese_hate_speech_hierarchical_classification.csv')

print("Categoria    Total")
print(f"No Hate Speech   {df.shape[0] - df[df.columns[1]].sum()}")
for i in range(1,80):
    nome = df.columns[i]
    soma = df[nome].sum()
    print(f"{nome}   {soma}")

