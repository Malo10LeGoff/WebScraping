# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:50:49 2020

@author: LENOVO
"""

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import math
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


df = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', engine = 'python', encoding = 'utf-8')
porter = PorterStemmer()

def cleaned_review(review):
    ### Keep only letters
    review = re.sub(r"[^a-zA-Z.!?']", ' ',review)
    review = review.lower()
    review = review.split()
    review_cleaned = []
    for word in review:
        if word not in set(stopwords.words('english')):
            review_cleaned.append(porter.stem(word))
    review = ' '.join(review_cleaned)
    return review

df = df.values

cleaned_df = [cleaned_review(text[0]) for text in df]

### Tokenisation
import tensorflow_datasets as tfds

tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    cleaned_df, target_vocab_size=2**12
)

data = [tokenizer.encode(text) for text in cleaned_df]

### Padding to put all the words to the same size

max_length = max([len(text) for text in data])
data = tf.keras.preprocessing.sequence.pad_sequences(data,value = 0, padding = 'post',maxlen = max_length)
data = np.asarray(data)
y = df[:,1]
y = np.reshape(y, (len(y),1))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data,y ,test_size = 0.2)

### The principle here is to transform each sentece into a matrix with rows corresponding to the words of the sentence and the columns the coordinates of the words in the embedded space
### Then we apply several 1D convolutions over this sentence to detect the features

class DCNN(tf.keras.Model):
    
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                          emb_dim)
        self.bigram = tf.keras.layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = tf.keras.layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.fourgram = tf.keras.layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
        self.pool = tf.keras.layers.GlobalMaxPool1D() # no training variable so we can
                                             # use the same layer for each
                                             # pooling step
        self.dense_1 = tf.keras.layers.Dense(units=FFN_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = tf.keras.layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = tf.keras.layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        
        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output
    
    
### Defining the hyperparameters
        
        
VOCAB_SIZE = tokenizer.vocab_size

EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2#len(set(train_labels))

DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NB_EPOCHS = 5

### Training phase

model = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


X_train = np.asarray(X_train, dtype = np.float)
y_train = np.asarray(y_train, dtype = np.float)

model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCHS)

X_test = np.asarray(X_test, dtype = np.float)
y_test = np.asarray(y_test, dtype = np.float)

model.evaluate(X_test, y_test)
        
        