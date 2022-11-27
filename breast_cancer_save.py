import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np


classifier = Sequential()
classifier.add(Dense(units = 8, activation='relu', kernel_initializer='normal', input_dim=30))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 8, activation='relu', kernel_initializer='normal'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation='sigmoid'))
classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

classifier.fit(X, y, batch_size=10, epochs=100)

classifier_json = classifier.to_json()
classifier.save_weights('classifier_weights.h5')

with open('classifier.json', 'w') as json_file:
    json_file.write(classifier_json)
