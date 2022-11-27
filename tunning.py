import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def CreateNetwork(optimizer, loss, kernel_initializer, activation, neurons):
    classifier = Sequential()
    classifier.add(Dense(units = neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = neurons, activation=activation, kernel_initializer=kernel_initializer))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1, activation='sigmoid'))
    classifier.compile(optimizer= optimizer, loss=loss, metrics=['binary_accuracy'])
    return classifier

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

classifier = KerasClassifier(build_fn=CreateNetwork)
parameters = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'optimizer': ['adam', 'sgd'],
    'loss': ['binary_crossentropy', 'hinge'],
    'kernel_initializer': ['random_uniform', 'normal'],
    'activation': ['relu', 'tanh'],
    'neurons': [16, 8]
}

gridSearch = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=5)
gridSearch = gridSearch.fit(X, y)
best_param = gridSearch.best_params_
best_precision = gridSearch.best_score_

print(f'Best params: {best_param} \nBest Precision = {best_precision}')
