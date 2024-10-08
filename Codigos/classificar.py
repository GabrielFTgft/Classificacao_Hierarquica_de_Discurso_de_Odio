import pandas as pd
df = pd.read_csv('C:\\Users\\marde\\OneDrive\\Documentos\\CC\\Mineracao_de_Dados\\Dataset_Paula_Fortuna\\portuguese_hate_speech_hierarchical_classification.csv')

novo_arquivo = 'hate_speech_classification.csv'
ndf = pd.DataFrame(columns=['text', 'class'])

for index, row in df.iterrows():
    if(row[1]): #hate speech
        linha = {'text': row[0], 'class': '0'}
        if(row[4]): #racism
            linha['class'] = '1'
            if(row[27]): #indiginous
                linha['class'] = '1.1'
            elif(row[47]): #white people
                linha['class'] = '1.2'
            elif(row[53]): #black people
                linha['class'] = '1.3'
        elif(row[7]): #origin
            linha['class'] = '2'
            if(row[14]): #asians
                linha['class'] = '1-2.1'
                if(row[19]): #chinese
                    linha['class'] = '1-2.1.1'
                elif(row[29]): #japaneses
                    linha['class'] = '1-2.1.2'
            elif(row[32]): #latins
                linha['class'] = '1-2.2'
                if(row[13]): #argentines
                    linha['class'] = '1-2.2.1'
                elif(row[35]): #mexicans
                    linha['class'] = '1-2.2.2'
                elif(row[51]): #venezuelans
                    linha['class'] = '1-2.2.3'
            elif(row[41]): #rural people
                linha['class'] = '2.3'
            elif(row[76]): #south americans
                linha['class'] = '2.4'
                if(row[75]): #brazilians
                    linha['class'] = '2.4.1'
                    if(row[37]): #nordestines
                        linha['class'] = '2.3-2.4.1.1'
                    elif(row[43]): #sertanejos
                        linha['class'] = '2.3-2.4.1.2'
            elif(row[73]): #africans
                linha['class'] = '2.5'
                if(row[52]): #angolans
                    linha['class'] = '2.5.1'
            elif(row[71]): #arabic
                linha['class'] = '2.6'
                if(row[21]): #egyptians
                    linha['class'] = '2.6.1'
                elif(row[28]): #iranians
                    linha['class'] = '2.6.2'
            elif(row[72]): #east europeans
                linha['class'] = '2.7'
                if(row[42]): #russians
                    linha['class'] = '2.7.1'
                elif(row[45]): #ucranians
                    linha['class'] = '2.7.2'
        elif(row[2]): #sexism
            linha['class'] = '3'
            if(row[61]): #men
                linha['class'] = '3.1'
            elif(row[68]): #transexuals
                linha['class'] = '3.2'
            elif(row[66]): #women
                linha['class'] = '3.3'
                if(row[64]): #trans women
                    linha['class'] = '3.2-3.3.1'
                elif(row[16]): #black women
                    linha['class'] = '1.3-3.3.2'
                elif(row[18]): #brazilian women 
                    linha['class'] = '3.3.3'
                elif(row[11]): #aborting women 
                    linha['class'] = '3.3.4'
                elif(row[65]): #travestis
                    linha['class'] = '3.3.5'
                elif(row[69]): #ugly women
                    linha['class'] = '8.2-3.3.6'
                elif(row[26]): #homeless women
                    linha['class'] = '11.7-3.3.7'
                elif(row[17]): #blond women
                    linha['class'] = '8-3.3.8'
                elif(row[70]): #thin women
                    linha['class'] = '8.1-3.3.9'
                elif(row[55]): #fat women
                    linha['class'] = '8.3-3.3.10'
                elif(row[49]): #old women
                    linha['class'] = '9.1-3.3.11'
                elif(row[23]): #football players women
                    linha['class'] = '11-3.3.12'
                elif(row[36]): #muslism women
                    linha['class'] = '5.3-3.3.13'
        elif(row[6]): #homophobia
            linha['class'] = '4'
            if(row[67]): #bissexuals
                linha['class'] = '4.1'
            elif(row[77]): #homossexuals
                linha['class'] = '4.2'
                if(row[60]): #lesbians
                    linha['class'] = '4.2.1'
                elif(row[57]): #gays
                    linha['class'] = '4.2.2'
        elif(row[8]): #religion
            linha['class'] = '5'
            if(row[59]): #islamits
                linha['class'] = '5.1'
            elif(row[30]): #jews
                linha['class'] = '5.2'
            elif(row[62]): #muslism
                linha['class'] = '5.3'
        elif(row[5]): #ideology
            linha['class'] = '6'
            if(row[56]): #feminists
                linha['class'] = '3-6.1'
                if(row[34]): #men feminist
                    linha['class'] = '3-6.1.1'
            elif(row[12]): #agnostic
                linha['class'] = '6.2'
            elif(row[33]): #left wing ideology
                linha['class'] = '6.3'
        elif(row[76]): #migrants
            linha['class'] = '7'
            if(row[58]): #immigrants
                linha['class'] = '7.1'
            elif(row[63]): #refugees
                linha['class'] = '7.2'
        elif(row[3]): #body
            linha['class'] = '8'
            if(row[78]): #thin people
                linha['class'] = '8.1'
            elif(row[50]): #ugly people
                linha['class'] = '8.2'
            elif(row[22]): #fat people
                linha['class'] = '8.3'
        elif(row[79]): #ageing
            linha['class'] = '9'
            if(row[38]): #old people
                linha['class'] = '9.1'
            elif(row[48]): #young people
                linha['class'] = '9.2'
        elif(row[9]): #health
            linha['class'] = '10'
            if(row[15]): #autists
                linha['class'] = '10.1'
            elif(row[54]): #disabled people
                linha['class'] = '10.2'
        elif(row[10]): #otherlyfestyles
            linha['class'] = '11'
            if(row[24]): #gamers
                linha['class'] = '11.1'
            elif(row[31]): #jornalists
                linha['class'] = '11.2'
            #????elif(row[]): #politicians
                #linha['class'] = '11.3'
            elif(row[46]): #vegetarians
                linha['class'] = '11.4'
            elif(row[39]): #polyamorous
                linha['class'] = '11.5'
            elif(row[40]): #poor people
                linha['class'] = '11.6'
            elif(row[25]): #homeless
                linha['class'] = '11.7'
            elif(row[20]): #criminals
                linha['class'] = '11.8'
            #????elif(row[]): #tattooed people
                #linha['class'] = '11.9'
            elif(row[44]): #street artisti
                linha['class'] = '11.10'
    else: #no hate speech
        linha = {'text': row[0], 'class': '-1'}
    ndf = pd.concat([ndf, pd.DataFrame([linha])], ignore_index=True)

ndf.to_csv(novo_arquivo, index=False)