'''
@author = Leonardo Pompeu
Description = Seguindo um curso da udemy, os códigos presentes nesse documento não são de minha autoria,
são códigos para compreensão e entendimento dos fundamentos de redes neurais artificiais em específico
estamos tratando de uma base de dados de classificação binária [0, 1] sendo que a saída da última camada
passará pela função de ativação da sigmoide onde valores maiores que 0.5 são classificados como 1 e menores
são classificados como 0

Apêndice: Noções importantes a serem consideradas:
- Documentação do keras: ferramenta muito importante na hora da construção da rede neural, principalmente
da seleção dos parâmetros.
- Validação cruzada: Principal método de validação da rede neural geralmente o K=10 segundo artigos científicos
não há necessidade de utilizar um valor de K muito alto.
- É possível obter os valores dos pesos atribuidos a cada camada utilizando o método get_weight()
- Existem vários tipos de métricas para o cálculo da acurácia e aqui estou utilizando 2, tanto um do sklearn
o accuracy_score e a confusion_matrix(Para ver a porcentagem de acerto de cada saída binária). E além desses
métodos do sklearn o próprio keras possui o método evaluate que permite o cálculo da acurácia, metodo que será
desenvolvido em modelos futuros.
'''
# ========================================================================
#                             IMPORTAÇÕES
# ========================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
# ========================================================================


# ========================================================================
#        LEITURA DA BASE DE DADOS E SEPARAÇÃO ENTRE TREINO E TESTE
# ========================================================================
X = pd.read_csv('entradas_breast.csv')
y = pd.read_csv('saidas_breast.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# ========================================================================

# ========================================================================
#     CRIAÇÃO DA REDE NEURAL, 2 CAMADAS OCULTAS E A CAMADA DE SAÍDA
# ========================================================================
classifier = Sequential()
classifier.add(Dense(units = 16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classifier.add(Dense(units = 16, activation='relu', kernel_initializer='random_uniform'))
classifier.add(Dense(units = 1, activation='sigmoid'))
# ========================================================================


# ========================================================================
# CONFIGURAÇÃO DO OTIMIZADOR, COMPILAÇÃO E FIT DOS DADOS NA REDE NEURAL
# ========================================================================
optimizer = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue=0.5)
classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
#classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
# ========================================================================


# ========================================================================
#        RETORNA OS PESOS CALCULADOS EM CADA CAMADA DA REDE NEURAL
# ========================================================================
weight = classifier.layers[0].get_weights()
weight1 = classifier.layers[1].get_weights()
weight2 = classifier.layers[2].get_weights()
# ========================================================================


# ========================================================================
#     UTILIZANDO AS MÉTRICAS DO SKLEARN PARA CÁLCULO DE ACURÁCIA
# ========================================================================
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
matriz = confusion_matrix(y_test, y_pred)
# ========================================================================


# ========================================================================
#     UTILIZANDO A MÉTRICA DO PRÓPRIO KERAS PARA CÁLCULO DE ACURÁCIA
# ========================================================================
result = classifier.evaluate(X_test, y_test)
print(f'Result using evaluate: {result}')
# ========================================================================