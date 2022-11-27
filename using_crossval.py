import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
'''
Pontos importantes: Foi utilizado um valor de Dropout de 0.2(20%) para evitar um Overfitting.
- É considerada uma boa prática a adição sempre do Dropout já que as redes neurais tem uma tendência
ao overfitting pela quantidade de parâmetros da mesma.
- Uma boa comparação a ser feita é entre o arquivo Main e esse arquivo utilizando tanto o CrossVal quanto o Dropout, mesmo
considerando que cada execução os resultados são ligeiramente diferentes, o algoritmo obteve uma melhora nos resultados
assim como no desvio padrão(Menos overfitting)
'''
def CreateNetwork():
    classifier = Sequential()
    classifier.add(Dense(units = 16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 16, activation='relu', kernel_initializer='random_uniform'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue=0.5)
    classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classifier

X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')

classifier = KerasClassifier(build_fn=CreateNetwork, epochs=100, batch_size=10)

results = cross_val_score(estimator=classifier, X = X, y=y, cv=10, scoring='accuracy')


'''
Tanto a média quanto o desvio padrão nos trazem uma acurácia mais próxima da real, principalmente
quando utilizamos o cross validation para fazer a divisão da rede neural.

Pontos importantes: Quanto maior o desvio padrão, maior é a chance de ocorrer o overfitting, que é o método
se tornar muito bom na base de treino porém quando novos valores são colocados a prova o resultado é ruim.
Ex de overfitting: Uma IA que aprende a dirigir um carro em uma pista e fica muito boa nessa pista específica,
mas quando trocamos ela de pista, a IA terá dificuldades.
'''
media = results.mean()
desvio = results.std()

print(f'Resultados: \n {results} \n Média: {media} \n Desvio Padrão: {desvio}')


