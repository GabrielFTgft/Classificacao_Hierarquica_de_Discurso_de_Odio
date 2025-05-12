import pandas as pd
from pathlib import Path
from paths import DATA_DIR

df = pd.read_csv(DATA_DIR / 'raw' /'portuguese_hate_speech_hierarchical_classification.csv')

file_path = Path(__file__).resolve().parent.parent / 'data' / 'preprocessed' / 'hierarchical_labels_dataset.csv'
ndf = pd.DataFrame(columns=['text', 'class'])

for index, row in df.iterrows():
    classes = []

    if(row[1]): #hate speech
        if(row[4]): #racism
            subRac = False #verificar se alguma subclasse foi adicionada

            if(row[27]): #indiginous
                classes.append('1.1')
                subRac = True
            if(row[47]): #white people
                classes.append('1.2') 
                subRac = True
            if(row[53]): #black people
                subRac = True
                if(row[16]): #black women
                    classes.append('1.3-3.3.1')
                else:
                    classes.append('1.3')
            if(row[14]): #asians
                subRac = True
                subAsians = False 
                if(row[19]): #chinese
                    subAsians = True
                    classes.append('1-2.1.1')
                if(row[29]): #japaneses
                    subAsians = True
                    classes.append('1-2.1.2')
                if not subAsians:
                    classes.append('1-2.1')
            if(row[32]): #latins
                subRac = True
                subLatins = False
                if(row[13]): #argentines
                    subLatins = True
                    classes.append('1-2.2.1')
                if(row[35]): #mexicans
                    subLatins = True
                    classes.append('1-2.2.2')
                if(row[51]): #venezuelans
                    subLatins = True
                    classes.append('1-2.2.3')
                if not subLatins:
                    classes.append('1-2.2')
            if not subRac:
                classes.append('1')
        if(row[7]): #origin
            subOrigin = False
            if(row[41]): #rural people
                subOrigin = True
                subRural = False
                if(row[37]): #nordestines
                    subRural = True
                    classes.append('2.1-2.2.1.1')
                if(row[43]): #sertanejos
                    subRural = True
                    classes.append('2.1-2.2.1.2')
                elif not subRural:
                    classes.append('2.1')
            if(row[76]): #south americans
                subOrigin = True
                if(row[75]) and not (row[37] or row[43]): #brazilians
                    classes.append('2.2.1')
                else:
                    classes.append('2.2')
            if(row[73]): #africans
                subOrigin = True
                if(row[52]): #angolans
                    classes.append('2.3.1')
                else:
                    classes.append('2.3')
            if(row[71]): #arabic
                subOrigin = True
                subArabic = False
                if(row[21]): #egyptians
                    subArabic = True
                    classes.append('2.4.1')
                if(row[28]): #iranians
                    subArabic = True
                    classes.append('2.4.2')
                elif not subArabic:
                    classes.append('2.4')
            if(row[72]): #east europeans
                subOrigin = True
                subEuropeans = False
                if(row[42]): #russians
                    subEuropeans = True
                    classes.append('2.5.1')
                if(row[45]): #ucranians
                    subEuropeans = True
                    classes.append('2.5.2')
                elif not subEuropeans:
                    classes.append('2.5')
            if not subOrigin and not row[32] and not row[14]:
                classes.append('2')
        if(row[2]): #sexism
            subSexism = False
            if(row[61]): #men
                subSexism = True
                classes.append('3.1')
            if(row[68]): #transexuals
                subSexism = True
                classes.append('3.2')
            if(row[66]): #women
                subSexism = True
                subWomen = False
                if(row[64]): #trans women
                    subWomen = True
                    classes.append('3.3.1')
                if(row[18]): #brazilian women 
                    subWomen = True
                    classes.append('3.3.2')
                if(row[11]): #aborting women 
                    subWomen = True
                    classes.append('3.3.3')
                if(row[65]): #travestis
                    subWomen = True
                    classes.append('3.3.4')
                if(row[69]): #ugly women
                    subWomen = True
                    classes.append('3.3-8.2.1')
                if(row[26]): #homeless women
                    subWomen = True
                    classes.append('3.3-11.7.1')
                if(row[17]): #blond women
                    subWomen = True
                    classes.append('3.3-8.1')
                if(row[70]): #thin women
                    subWomen = True
                    classes.append('3.3-8.1.1')
                if(row[55]): #fat women
                    subWomen = True
                    classes.append('3.3-8.3.1')
                if(row[49]): #old women
                    subWomen = True
                    classes.append('3.3-9.1.1')
                if(row[23]): #football players women
                    subWomen = True
                    classes.append('3.3-11.1')
                if(row[36]): #muslism women
                    subWomen = True
                    classes.append('3.3-5.3.1')
                elif not subWomen and not row[16]:
                    classes.append('3.3')
            if(row[56]): #feminists
                subSexism = True
                if(row[34]): #men feminist
                    classes.append('3-6.1.1')
                else:
                    classes.append('3-6.1')
            elif not subSexism:
                classes.append('3')
        if(row[6]): #homophobia
            subHomop = False
            if(row[67]): #bissexuals
                subHomop = True
                classes.append('4.1')
            if(row[77]): #homossexuals
                subHomop = True
                subHomos = False
                if(row[60]): #lesbians
                    subHomos = True
                    classes.append('4.2.1')
                if(row[57]): #gays
                    subHomos = True
                    classes.append('4.2.2')
                elif not subHomos:
                    classes.append('4.2')
            elif not subHomop:
                classes.append('4')
        if(row[8]): #religion
            subReligion = False
            if(row[59]): #islamits
                subReligion = True
                classes.append('5.1')
            if(row[30]): #jews
                subReligion = True
                classes.append('5.2')
            if(row[62]): #muslism
                subReligion = True
                classes.append('5.3')
            elif not subReligion and not row[36]:
                classes.append('5')
        if(row[5]): #ideology
            subIdeo = False
            if(row[12]): #agnostic
                subIdeo = True
                classes.append('6.1')
            if(row[33]): #left wing ideology
                subIdeo = True
                classes.append('6.2')
            elif not subIdeo and not row[56]:
                classes.append('6')
        if(row[76]): #migrants
            subMig = False
            if(row[58]): #immigrants
                subMig = True
                classes.append('7.1')
            if(row[63]): #refugees
                subMig = True
                classes.append('7.2')
            elif not subMig:
                classes.append('7')
        if(row[3]): #body
            subBody = False
            if(row[78]): #thin people
                subBody = True
                if not row[70]:
                    classes.append('8.1')
            if(row[50]): #ugly people
                subBody = True
                if not row[69]:
                    classes.append('8.2')
            if(row[22]): #fat people
                subBody = True
                if not row[55]:
                    classes.append('8.3')
            elif not subBody and not row[17]:
                classes.append('8')
        if(row[79]): #ageing
            subAge = False
            if(row[38]): #old people
                subAge = True
                if not row[49]:
                    classes.append('9.1')
            if(row[48]): #young people
                subAge = True
                classes.append('9.2')
            elif not subAge:
                classes.append('9')
        if(row[9]): #health
            subHeal = False
            if(row[15]): #autists
                subHeal = True
                classes.append('10.1')
            if(row[54]): #disabled people
                subHeal = True
                classes.append('10.2')
            elif not subHeal:
                classes.append('10')
        if(row[10]): #otherlyfestyles
            subOther = False
            if(row[24]): #gamers
                subOther = True
                classes.append('11.1')
            if(row[31]): #jornalists
                subOther = True
                classes.append('11.2')
            #????elif(row[]): #politicians
                #classes.append('11.3'
            if(row[46]): #vegetarians
                subOther = True
                classes.append('11.4')
            if(row[39]): #polyamorous
                subOther = True
                classes.append('11.5')
            if(row[40]): #poor people
                subOther = True
                classes.append('11.6')
            if(row[25]): #homeless
                subOther = True
                if not row[26]:
                    classes.append('11.7')
            if(row[20]): #criminals
                subOther = True
                classes.append('11.8')
            #????elif(row[]): #tattooed people
                #classes.append('11.9'
            if(row[44]): #street artisti
                subOther = True
                classes.append('11.10')
            elif not subOther and not row[23]:
                classes.append('11')
    else:
        classes.append('-1') 

    linha = {'text': row[0], 'class': '@'.join(classes)}
    ndf = pd.concat([ndf, pd.DataFrame([linha])], ignore_index=True)

ndf.to_csv(file_path, index=False)
print("Classificacao concluida")