#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: sid
"""
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout




## Reading the dataset
df = pd.read_csv('Practical/dataset/fake-news/train.csv')
df.head()


## We will drop the Nan values now
df = df.dropna()

## Getting the independent and dependent features
X = df.drop('label', axis = 1 )
y = df['label']


content = X.copy()
content.reset_index(inplace=True)



X.shape


import tensorflow as tf



## In order to access the sentences in vectors. We will preprocess the data. 


## First we will remove the stop words. 
import nltk
import re
from nltk.corpus import stopwords


## Preprocessing the data to solve the problem of sparse matrix

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

voc_size = 5000

## Corpus will be our neww variable for storing everyhting
corpus = [] 

for i in range (0,len(content)):
    
    words_rev = re.sub('[^a-zA-Z]' , " ", content['title'][i])
    words_rev = words_rev.lower()
    
    ## .split() converts a string into a list
    
    words_rev = words_rev.split()
    
    
    ##Stemming those woirds in words_rev that are not a part of stop words
    words_rev = [ps.stem(word) for word in words_rev if not word in stopwords.words('english')]
    
    ## Adding the stemmed words back to words_rev and adding it to corpus
    words_rev = ' '.join(words_rev)
    corpus.append(words_rev)
    
    
    ## Printing corpus to see the stemmed words


    ## Vectorizing the words into array 
    ## Creatinf the embedded representation of it
    
o_repr = [one_hot(words,voc_size) for words in corpus]
    
    
new_length = 20
embedded_docs = pad_sequences(o_repr, padding = 'pre', maxlen = new_length)
    
    
    
    ## We will add dropout layer in between to prevent overfitting 
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=new_length))
model.add(Dropout(0.2))
model.add(LSTM(100) )
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
   
    
print(model.summary())


X_f = np.array(embedded_docs)
y_f = np.array(y)

## Now we will convert the data into Training and test data
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X_f, y_f, test_size= 0.30, random_state=42)

##tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=30, batch_size=64)


y_pred=model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix



confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)





    
            