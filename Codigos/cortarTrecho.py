import pandas as pd
df = pd.read_csv('hate_speech_classification.csv')

trecho_30_tweets =  df.head(30)
trecho_30_tweets.to_csv('trecho_tweets.csv', index=True)