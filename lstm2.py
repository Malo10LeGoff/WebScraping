# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:22:42 2020

@author: LENOVO
"""

import tensorflow as tf
import keras
import numpy as np 
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
porter = PorterStemmer()

df = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', engine = 'python', encoding = 'utf-8')

### Cleaning text

def cleaning_review(text):
    review = re.sub(r"[^a-zA-Z.!?']", ' ',text)
    review = review.lower()
    review = review.split()
    cleaned_review = []
    for word in review:
        if word not in set(stopwords.words('english')):
            print(porter.stem(word))
            cleaned_review.append(porter.stem(word))
    cleaned_review = ' '.join(cleaned_review)
    return cleaned_review

df_cleaned = [cleaning_review(review[0]) for review in df.values]
print(df_cleaned)

### We try a second approach here, transforming the sentence into a matrix were each row is a word and each column the coordinates of this row in the embedded space
### Then we feed this matrix into a LSTM layer. But first we have to tokenize and then padd our dataset

### Tokenizer

import tensorflow_datasets as tfds

tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    df_cleaned, target_vocab_size=2**12
)

data = [tokenizer.encode(text) for text in df_cleaned]
max_len = max([len(data[i]) for i in range(0, len(data))])

### Padding

data = tf.keras.preprocessing.sequence.pad_sequences(data, value = 0, maxlen = max_len, padding = 'post')
y = df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.2)


embed_dim = 128
VOCAB_SIZE = tokenizer.vocab_size

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(output_dim = embed_dim, input_dim = VOCAB_SIZE))  ### each word of the sentence is a timestep
### And each word passes through the embedding layer so the size of the input is equal to the size of the sentence. That's normal because 
### To train the embedding layer input a one hot encoded vector.


model.add(tf.keras.layers.LSTM(units = 128, return_sequences = True))
model.add(tf.keras.layers.Dropout(rate = 0.1))


model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, batch_size = 32)

model.summary()

from sklearn.metrics import confusion_matrix

y_pred = model.evaluate(X_test,y_test)
