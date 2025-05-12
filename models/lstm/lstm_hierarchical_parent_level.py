import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from scripts import binary_classifier, multilabel_classifier
from collections import defaultdict
from paths import DATA_DIR
from scripts import pre_text_processing

def gerar_relatorio_classificacao(nome, y_test, y_pred):
    print(nome)
    print(classification_report(y_test, y_pred))

def adicionarClassificacao(y_test_classes_preditas,X_test, classe):
    for value in X_test:
        y_test_classes_preditas[tuple(value)].append(classe)

dados = pre_text_processing.preprocess_text(DATA_DIR / 'preprocessed' / 'dataset_hierarchical_28.csv')

with open(DATA_DIR / 'preprocessed' /'embeddings_index.pkl', "rb") as f:
    embeddings_index = pickle.load(f)

max_len = 100 
tokenizer = Tokenizer() # Inicializa o tokenizer
tokenizer.fit_on_texts(dados['text'])  # Ajusta o tokenizer aos seus tokens


word_index = tokenizer.word_index
tam_voc = len(word_index) + 1

sequences = tokenizer.texts_to_sequences(dados['text'])  # Converter os textos para índices
data = pad_sequences(sequences, maxlen=max_len) # Ajustar o comprimento das sequências

labels = dados.drop(columns=['text'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

y_test_classes_preditas = defaultdict(list)
for value in X_test:
    y_test_classes_preditas[tuple(value)] = [] 

modelo_hate_speech, _ = binary_classifier.treinar_classificador_binario(X_train, X_test, y_train['Hate.speech'], y_test['Hate.speech'], tam_voc, word_index, embeddings_index)

y_pred = (modelo_hate_speech.predict(X_test) > 0.5).astype("int32")
gerar_relatorio_classificacao("Relatório de classificação - Hate Speech", y_test['Hate.speech'], y_pred)

f1 = f1_score(y_test['Hate.speech'], y_pred)
print("F1 Score:", f1)

hate_speech_indices = np.where(y_pred.flatten() == 1)[0]
X_test_hate_speech = X_test[hate_speech_indices]
y_test_hate_speech = y_test.iloc[hate_speech_indices]

adicionarClassificacao(y_test_classes_preditas,X_test_hate_speech,'Hate.speech')

X_train_hate_speech = X_train[y_train['Hate.speech'] == 1]
y_train_hate_speech = y_train[y_train['Hate.speech'] == 1]

filhos_hate_speech = ['Sexism','Body','Racism','Ideology','Homophobia','Origin','Religion','OtherLifestyle','Migrants']

lista_classes= []
for index, row in y_train_hate_speech.iterrows():
    classes = [row[col] for col in filhos_hate_speech if col in row]
    lista_classes.append(classes)
y_train_filhos = lista_classes

lista_classes= []
for index, row in y_test_hate_speech.iterrows():
    classes = [row[col] for col in filhos_hate_speech if col in row]
    lista_classes.append(classes)
y_test_filhos = lista_classes

num_classes = len(filhos_hate_speech)

y_train_filhos = np.array(y_train_filhos)
y_test_filhos = np.array(y_test_filhos)

modelo_filhos_hate_speech, _ = multilabel_classifier.treinar_classificador_multiclasse(X_train_hate_speech, X_test_hate_speech, y_train_filhos, y_test_filhos, tam_voc, num_classes, word_index, embeddings_index)


y_pred_filhos = (modelo_filhos_hate_speech.predict(X_test_hate_speech) > 0.5).astype("int32")
gerar_relatorio_classificacao("Relatório de classificação - Filhos de Hate Speech: Sexism, Body, Racism, Ideology, Homophobia, Origin, Religion, OtherLifestyle, Migrants", y_test_filhos, y_pred_filhos)
f1 = f1_score(y_test_filhos, y_pred_filhos, average='samples')
print("f1 Score Weighted:", f1_score(y_test_filhos, y_pred_filhos, average='weighted'))
# Dicionário para armazenar os y_test específicos para cada classe
y_test_especificos = {filho: [] for filho in filhos_hate_speech}

# Para cada exemplo no conjunto de teste
for i, row in enumerate(y_pred_filhos):
    # Para cada filho de hate speech, verifica se o valor é 1
    for j, predicted in enumerate(row):
        if predicted == 1:
            nome_filho = filhos_hate_speech[j]
            y_test_especificos[nome_filho].append(y_test_filhos[i])  

# Classificação para a categoria Sexism com 4 filhos
if len(y_test_especificos['Sexism']) > 0:
    # Filtra os exemplos de 'Sexism' nos dados de treino e teste
    X_train_sexism = X_train_hate_speech[y_train_hate_speech['Sexism'] == 1]
    y_train_sexism = y_train_hate_speech[y_train_hate_speech['Sexism'] == 1][['Men', 'Transexuals', 'Women', 'Feminists']]

    # Filtra os índices onde a posição 5 (Sexism) é 1 em y_pred
    indices_sexism = [i for i, row in enumerate(y_pred_filhos) if row[0] == 1]

    # Filtra X_test e y_test_filhos com base nesses índices
    X_test_sexism = X_test_hate_speech[indices_sexism]

    adicionarClassificacao(y_test_classes_preditas,X_test_sexism,'Sexism')

    # Extrai as classes específicas "Men", "Transexuals", "Women" e "Feminists" para y_train e y_test
    y_train_sexism = np.array(y_train_sexism)
    y_test_sexism = np.array(y_test_hate_speech[['Men', 'Transexuals', 'Women', 'Feminists']].iloc[indices_sexism].values)

    # Treinamento do modelo para as classes de sexism
    modelo_sexism, _ = multilabel_classifier.treinar_classificador_multiclasse(X_train_sexism, X_test_sexism, y_train_sexism, y_test_sexism, tam_voc, 4, word_index, embeddings_index)

    # Predição e relatório de classificação
    y_pred_sexism = (modelo_sexism.predict(X_test_sexism) > 0.5).astype("int32")
    gerar_relatorio_classificacao("Relatório de classificação - Men, Transexuals, Women e Feminists", y_test_sexism, y_pred_sexism)

    indices_men = [i for i, row in enumerate(y_pred_sexism) if row[0] == 1]
    indices_transexual = [i for i, row in enumerate(y_pred_sexism) if row[1] == 1]
    indices_women = [i for i, row in enumerate(y_pred_sexism) if row[2] == 1]
    indices_feminists = [i for i, row in enumerate(y_pred_sexism) if row[3] == 1]

    if indices_men:
        X_test_men = X_test_sexism[indices_men]
        adicionarClassificacao(y_test_classes_preditas, X_test_men, 'Men')
    
    if indices_transexual:
        X_test_transexual = X_test_sexism[indices_transexual]
        adicionarClassificacao(y_test_classes_preditas, X_test_transexual, 'Transexuals')
        
    if indices_feminists:
        X_test_feminists = X_test_sexism[indices_feminists]
        adicionarClassificacao(y_test_classes_preditas, X_test_feminists, 'Feminists')

    if indices_women:
        X_test_women = X_test_sexism[indices_women]
        adicionarClassificacao(y_test_classes_preditas, X_test_women, 'Women')

        X_train_women = X_train_hate_speech[y_train_hate_speech['Women'] == 1]
        y_train_women = y_train_hate_speech[y_train_hate_speech['Women'] == 1][['Ugly.women', 'Trans.women', 'Fat.women']]

        y_train_women = np.array(y_train_women)
        y_temp = y_test_hate_speech[['Ugly.women', 'Trans.women', 'Fat.women']].iloc[indices_sexism].values
        y_test_women = np.array(y_temp[indices_women])
        #y_test_women = np.array(y_test_hate_speech[['Ugly.women', 'Trans.women', 'Fat.women']].iloc[indices_women].values)

        modelo_women_filhos, _ = multilabel_classifier.treinar_classificador_multiclasse(X_train_women, X_test_women, y_train_women, y_test_women, tam_voc, 3, word_index, embeddings_index)

        y_pred_women_filhos = (modelo_women_filhos.predict(X_test_women) > 0.5).astype("int32")

        indices_ugly_women = [i for i, row in enumerate(y_pred_women_filhos) if row[0] == 1]
        indices_trans_women = [i for i, row in enumerate(y_pred_women_filhos) if row[1] == 1]
        indices_fat_women = [i for i, row in enumerate(y_pred_women_filhos) if row[2] == 1]

        if indices_ugly_women:
            X_test_ugly_women = X_test_women[indices_ugly_women]
            adicionarClassificacao(y_test_classes_preditas, X_test_ugly_women, 'Ugly.women')
        
        if indices_trans_women:
            X_test_trans_women = X_test_women[indices_trans_women]
            adicionarClassificacao(y_test_classes_preditas, X_test_trans_women, 'Trans.women')
        
        if indices_fat_women:
            X_test_fat_women = X_test_women[indices_fat_women]
            adicionarClassificacao(y_test_classes_preditas, X_test_fat_women, 'Fat.women')
        
        gerar_relatorio_classificacao("Relatório de classificação - Ugly.women, Trans.women e Fat.women", y_test_women, y_pred_women_filhos)

if len(y_test_especificos['Body']) > 0:
    X_train_body = X_train_hate_speech[(y_train_hate_speech['Body'] == 1)]
    y_train_body = y_train_hate_speech[(y_train_hate_speech['Body'] == 1)][['Ugly.people', 'Fat.people']]

    indices_body = [i for i, row in enumerate(y_pred_filhos) if row[1] == 1]
    
    X_test_body = X_test_hate_speech[indices_body]

    adicionarClassificacao(y_test_classes_preditas, X_test_body, 'Body')

    y_train_body = np.array(y_train_body)
    y_test_body = np.array(y_test_hate_speech[['Ugly.people', 'Fat.people']].iloc[indices_body].values)

    modelo_body, _ = multilabel_classifier.treinar_classificador_multiclasse(X_train_body, X_test_body, y_train_body, y_test_body, tam_voc, 2, word_index, embeddings_index)

    y_pred_body = (modelo_body.predict(X_test_body) > 0.5).astype("int32")

    indices_ugly_people = [i for i, row in enumerate(y_pred_body) if row[0] == 1]
    indices_fat_people = [i for i, row in enumerate(y_pred_body) if row[1] == 1]

    if indices_ugly_people:
        X_test_ugly_people = X_test_body[indices_ugly_people]
        adicionarClassificacao(y_test_classes_preditas, X_test_ugly_people, 'Ugly.people')

    if indices_fat_people:
        X_test_fat_people = X_test_body[indices_fat_people]
        adicionarClassificacao(y_test_classes_preditas, X_test_fat_people, 'Fat.people')

    gerar_relatorio_classificacao("Relatório de classificação - Ugly.people e Fat.people", y_test_body, y_pred_body)

    #Falta o binário de Ugly Women e Fat Women

if len(y_test_especificos['Racism']) > 0:
    X_train_racism = X_train_hate_speech[y_train_hate_speech['Racism'] == 1]
    y_train_racism = y_train_hate_speech[y_train_hate_speech['Racism'] == 1]

    indices_racism = [i for i, row in enumerate(y_pred_filhos) if row[2] == 1]

    # Filtra X_test e y_test_filhos com base nesses índices
    X_test_racism = X_test_hate_speech[indices_racism]

    adicionarClassificacao(y_test_classes_preditas, X_test_racism, 'Racism')

    y_train_racism = y_train_racism['Black.people']
    y_test_racism = y_test_hate_speech['Black.people'].iloc[indices_racism]

    modelo_black_people, _ = binary_classifier.treinar_classificador_binario(X_train_racism, X_test_racism, y_train_racism, y_test_racism, tam_voc, word_index, embeddings_index)
    y_pred_black_people = (modelo_black_people.predict(X_test_racism) > 0.5).astype("int32")

    indices_black_people = [i for i, row in enumerate(y_pred_black_people) if row == 1]

    if indices_black_people:
        X_test_black_people = X_test_racism[indices_black_people]
        adicionarClassificacao(y_test_classes_preditas, X_test_black_people, 'Black.people')
    
    gerar_relatorio_classificacao("Relatório de classificação - Black.people", y_test_racism, y_pred)

if len(y_test_especificos['Ideology']) > 0:
    X_train_ideology = X_train_hate_speech[(y_train_hate_speech['Ideology'] == 1)]
    y_train_ideology = y_train_hate_speech[(y_train_hate_speech['Ideology'] == 1)][['Feminists', 'Left.wing.ideology']]

    indices_ideology = [i for i, row in enumerate(y_pred_filhos) if row[3] == 1]

    X_test_ideology = X_test_hate_speech[indices_ideology]
    adicionarClassificacao(y_test_classes_preditas, X_test_ideology, 'Ideology')

    y_train_ideology = np.array(y_train_ideology)
    y_test_ideology = np.array(y_test_hate_speech[['Feminists', 'Left.wing.ideology']].iloc[indices_ideology].values)

    modelo_ideology, _ = multilabel_classifier.treinar_classificador_multiclasse(X_train_ideology, X_test_ideology, y_train_ideology, y_test_ideology, tam_voc, 2, word_index, embeddings_index)

    y_pred_ideology = (modelo_ideology.predict(X_test_ideology) > 0.5).astype("int32")

    indices_feminists = [i for i, row in enumerate(y_pred_ideology) if row[0] == 1]
    indices_left_wing_ideology = [i for i, row in enumerate(y_pred_ideology) if row[1] == 1]

    if indices_feminists:    
        X_test_feminists = X_test_ideology[indices_feminists]
        adicionarClassificacao(y_test_classes_preditas, X_test_feminists, 'Feminists')

    if indices_left_wing_ideology:
        X_test_left_wing_ideology = X_test_ideology[indices_left_wing_ideology]
        adicionarClassificacao(y_test_classes_preditas, X_test_left_wing_ideology, 'Left.wing.ideology')

    gerar_relatorio_classificacao("Relatório de classificação - Feminists e Left.wing.ideology", y_test_ideology, y_pred_ideology)

if len(y_test_especificos['Homophobia']) > 0:
    X_train_homophobia = X_train_hate_speech[y_train_hate_speech['Homophobia'] == 1]
    y_train_homophobia = y_train_hate_speech[y_train_hate_speech['Homophobia'] == 1]
    
    indices_homophobia = [i for i, row in enumerate(y_pred_filhos) if row[4] == 1]

    X_test_homophobia = X_test_hate_speech[indices_homophobia]
    adicionarClassificacao(y_test_classes_preditas, X_test_homophobia, 'Homophobia')

    y_train_homophobia = y_train_homophobia['Homossexuals'] 
    y_test_homophobia = y_test_hate_speech['Homossexuals'].iloc[indices_homophobia]

    modelo_homossexuals, _ = binary_classifier.treinar_classificador_binario(X_train_homophobia, X_test_homophobia, y_train_homophobia, y_test_homophobia, tam_voc, word_index, embeddings_index)

    y_pred_homossexuals = (modelo_homossexuals.predict(X_test_homophobia) > 0.5).astype("int32")
    gerar_relatorio_classificacao("Relatório de classificação - Homossexuals", y_test_homophobia, y_pred_homossexuals)

    # Filtra os índices onde a posição de Homossexuals (supondo que esteja na posição 0) é 1 em y_pred
    indices_homossexuals = [i for i, row in enumerate(y_pred_homossexuals) if row == 1]

    X_test_homossexuals = X_test_homophobia[indices_homossexuals]
    adicionarClassificacao(y_test_classes_preditas, X_test_homossexuals, 'Homossexuals')

    # Filtra as classes "Lesbians" e "Gays" para y_train e y_test
    X_train_homossexuals = X_train_hate_speech[y_train_hate_speech['Homossexuals'] == 1]
    y_train_homossexuals = y_train_hate_speech[y_train_hate_speech['Homossexuals'] == 1][['Lesbians', 'Gays']]

    y_train_homossexuals = np.array(y_train_homossexuals)
    y_temp2 = y_test_hate_speech[['Lesbians', 'Gays']].iloc[indices_homophobia].values
    y_test_homossexuals = np.array(y_temp2[indices_homossexuals])

    # Treinamento do modelo para as classes Lesbians e Gays
    modelo_homossexuals_filhos, _ = multilabel_classifier.treinar_classificador_multiclasse(X_train_homossexuals, X_test_homossexuals, y_train_homossexuals, y_test_homossexuals, tam_voc, 2, word_index, embeddings_index)

    # Predição para Lesbians e Gays
    y_pred_homossexuals_filhos = (modelo_homossexuals_filhos.predict(X_test_homossexuals) > 0.5).astype("int32")

    indices_lesbians = [i for i, row in enumerate(y_pred_homossexuals_filhos) if row[0] == 1]
    indices_gays = [i for i, row in enumerate(y_pred_homossexuals_filhos) if row[1] == 1]

    if indices_lesbians:
        X_test_lesbians = X_test_homossexuals[indices_lesbians]
        adicionarClassificacao(y_test_classes_preditas, X_test_lesbians, 'Lesbians')

    if indices_gays:
        X_test_gays = X_test_homossexuals[indices_gays]
        adicionarClassificacao(y_test_classes_preditas, X_test_gays, 'Gays')

    gerar_relatorio_classificacao("Relatório de classificação - Lesbians e Gays", y_test_homossexuals, y_pred_homossexuals_filhos)

if len(y_test_especificos['Origin']) > 0:
    X_test_origin = X_test_hate_speech[y_test_hate_speech['Origin'] == 1]
    adicionarClassificacao(y_test_classes_preditas, X_test_origin, 'Origin')

if len(y_test_especificos['Religion']) > 0:
    # Filtra os exemplos de 'Religion' nos dados de treino e teste
    X_train_religion = X_train_hate_speech[(y_train_hate_speech['Religion'] == 1)]
    y_train_religion = y_train_hate_speech[(y_train_hate_speech['Religion'] == 1)][['Islamits', 'Muslims']]

    # Filtra os índices onde a posição 6 (Religion) é 1 em y_pred
    indices_religion = [i for i, row in enumerate(y_pred_filhos) if row[6] == 1]
    
    # Filtra X_test e y_test_filhos com base nesses índices
    X_test_religion = X_test_hate_speech[indices_religion]
    adicionarClassificacao(y_test_classes_preditas, X_test_religion, 'Religion')

    # Extrai as classes específicas "Islamits" e "Muslims" para y_train e y_test
    y_train_religion = np.array(y_train_religion)
    y_test_religion = np.array(y_test_hate_speech[['Islamits', 'Muslims']].iloc[indices_religion].values)

    # Treinamento do modelo para as classes Islamits e Muslims
    modelo_religion, _ = multilabel_classifier.treinar_classificador_multiclasse(X_train_religion, X_test_religion, y_train_religion, y_test_religion, tam_voc, 2, word_index, embeddings_index)

    y_pred_religion = (modelo_religion.predict(X_test_religion) > 0.5).astype("int32")

    indices_islamits = [i for i, row in enumerate(y_pred_religion) if row[0] == 1]
    indices_muslims = [i for i, row in enumerate(y_pred_religion) if row[1] == 1]

    if indices_islamits:
        X_test_islamits = X_test_religion[indices_islamits] 
        adicionarClassificacao(y_test_classes_preditas, X_test_islamits, 'Islamits')

    if indices_muslims:
        X_test_muslims = X_test_religion[indices_muslims] 
        adicionarClassificacao(y_test_classes_preditas, X_test_muslims, 'Muslims')

    gerar_relatorio_classificacao("Relatório de classificação - Islamits e Muslims", y_test_religion, y_pred_religion)
 
if len(y_test_especificos['OtherLifestyle']) > 0:
    X_test_other_lifestyle = X_test_hate_speech[y_test_hate_speech['OtherLifestyle'] == 1]
    adicionarClassificacao(y_test_classes_preditas, X_test_other_lifestyle, 'OtherLifestyle')

if len(y_test_especificos['Migrants']) > 0:
    X_train_migrants = X_train_hate_speech[(y_train_hate_speech['Migrants'] == 1)]
    y_train_migrants = y_train_hate_speech[(y_train_hate_speech['Migrants'] == 1)][['Immigrants', 'Refugees']]

    indices_migrants = [i for i, row in enumerate(y_pred_filhos) if row[8] == 1]
    
    X_test_migrants = X_test_hate_speech[indices_migrants]
    adicionarClassificacao(y_test_classes_preditas, X_test_migrants, 'Migrants')

    y_train_migrants = np.array(y_train_migrants)
    y_test_migrants = np.array(y_test_hate_speech[['Immigrants', 'Refugees']].iloc[indices_migrants].values)

    modelo_migrants, _ = multilabel_classifier.treinar_classificador_multiclasse(X_train_migrants, X_test_migrants, y_train_migrants, y_test_migrants, tam_voc, 2, word_index, embeddings_index)

    y_pred_migrants = (modelo_migrants.predict(X_test_migrants) > 0.5).astype("int32")

    indices_immigrants = [i for i, row in enumerate(y_pred_migrants) if row[0] == 1]
    indices_refugees = [i for i, row in enumerate(y_pred_migrants) if row[1] == 1]

    if indices_immigrants:
        X_test_immigrants = X_test_migrants[indices_immigrants]
        adicionarClassificacao(y_test_classes_preditas, X_test_immigrants, 'Immigrants')

    if indices_refugees:
        X_test_refugees = X_test_migrants[indices_refugees]
        adicionarClassificacao(y_test_classes_preditas, X_test_refugees, 'Refugees')

    gerar_relatorio_classificacao("Relatório de classificação - Immigrants e Refugees", y_test_migrants, y_pred_migrants)

file_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'preprocessed' / 'lstm_hierarchical_parent_level_predictions.txt'

with open(file_path, "w") as file:
    lista = []
    for chave, lista_classificacao in y_test_classes_preditas.items():
        lista.append(lista_classificacao)
    chaves = list(y_test_classes_preditas.keys())
    text = tokenizer.sequences_to_texts(chaves)
    for c, l in zip(text, lista):
        file.write(f"{c}:{l}\n")