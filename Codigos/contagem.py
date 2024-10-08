import pandas as pd
df = pd.read_csv('C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Dataset_Paula_Fortuna\\portuguese_hate_speech_hierarchical_classification.csv')

print("Categoria    Total")
print(f"No Hate Speech   {df.shape[0] - df[df.columns[1]].sum()}")
for i in range(1,80):
    nome = df.columns[i]
    soma = df[nome].sum()
    print(f"{nome}   {soma}")


# tweets = pd.read_csv('C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Arquivos_Gerados\\dataset_noStopWords.csv')

# max_length = max(tweets['tokens'].apply(eval).apply(len))  # Calcula o comprimento máximo
# print(f"A maior sequência de tokens tem comprimento: {max_length}")

