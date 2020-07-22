# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:48:21 2020

@author: LENOVO
"""

import numpy as np
import random
import pandas as pd

### We load our scraped dataset and fill the missing values with 1

df = pd.read_csv("reviews.csv", delimiter = '::',engine = 'python', encoding = 'utf-8')
df2 = df.values
df2[df2[:,1]=='None',1] = 1
df2 = df2[df2[:,1] != 'None1']

### We must transform all of our labels into integers for our model
for i in range(0,len(df2)):
    if type(df2[i,1]) == str:
        df2[i,1] = int(df2[i,1])
        print('String detected')
        print(i)
        
### Preprocessing of the data : cleaning the text

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
porter.stem('played')
corpus = []

for review in df2:
    text = review[0]
    cleaned_review = []
    text = text.lower()
    text = text.split()
    #print("text :" + str(text))
    for words in text:
        if words not in set(stopwords.words('french')):
            cleaned_review.append(porter.stem(words))
    cleaned_review = ' '.join(cleaned_review)
    #print("final review : " + str(cleaned_review))
    corpus.append(cleaned_review)


### This cleaning can lead to errors sometimes, In our case we have a review "il ne faut pas le rater" so something positive
### turned into "faut rater" after stopwords and stemming so the meaning changed
    
### Creation of the bag of words models because after the cleaning we have a lot less words
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)  ## only consider the 1500 more frequent words
X = cv.fit_transform(corpus).toarray()
y = df2[:,1]
y=y.astype('int')

### Split between training and testing set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
       
### Model and training

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

### Evaluation

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



