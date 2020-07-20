# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:29:12 2020

@author: LENOVO
"""

#### LSTM 

### Imports 

import pandas as pd
import numpy as np 
import tensorflow as tf
import keras

### Preprocessing the data (timesteps, number of inputs)

df = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', engine = 'python', encoding = 'utf-8')
y = df.iloc[:,1].values
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
corpus = []

df = df.values
for text in df:
    review = text[0]
    review = review.lower()
    review = review.split()
    review_cleaned = []
    for word in review:
        if word not in set(stopwords.words('english')):
            review_cleaned.append(stemmer.stem(word))
    review_cleaned = ' '.join(review_cleaned)
    corpus.append(review_cleaned)
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 300)
X = cv.fit_transform(corpus).toarray()
print(X.shape)
X = np.reshape(X,(X.shape[0],X.shape[1],1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
### Building of the model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1],1)))
model.add(tf.keras.layers.Dropout(rate = 0.1))

model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(rate = 0.1))

model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(rate = 0.1))

model.add(tf.keras.layers.LSTM(units = 32, return_sequences = True))

model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 100, batch_size = 32)

model.summary()

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)











