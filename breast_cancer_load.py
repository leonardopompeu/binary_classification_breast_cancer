import numpy as np
from keras.models import model_from_json
import pandas as pd

arquivo = open('classifier.json', 'r')
estrutura = arquivo.read()
arquivo.close()

classifier = model_from_json(estrutura)
classifier.load_weights('classifier_weights.h5')

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])
previsao = classifier.predict(novo)

previsao = (previsao > 0.5)

print(previsao)

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
results = classifier.evaluate(X, y)

print(results)