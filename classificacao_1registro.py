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

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])
previsao = classifier.predict(novo)

previsao = (previsao > 0.5)
