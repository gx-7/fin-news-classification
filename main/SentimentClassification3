import numpy as np
import pandas as pd
import unicodedata
import seaborn as sns

from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

%matplotlib inline

data = pd.read_csv("data/data.csv", index_col=0, encoding = "UTF-8")

def sentimentanalysis(param):
    sentimentAnalyzer = SentimentIntensityAnalyzer()
    for date in param.index:
         phrase = unicodedata.normalize('NFKD', param.loc[date, 'News'])
         score = sentimentAnalyzer.polarity_scores(phrase) 
         param.set_value(date, 'pos', score['pos'])
         param.set_value(date, 'neu', score['neu'])
         param.set_value(date, 'neg', score['neg'])
    return param
    
sentis = sentimentanalysis(data)

sentis.drop(['News'], axis=1, inplace=True)

plot(sentis)
