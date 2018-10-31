# -*- coding: utf-8 -*-
"""
Ciência de Dados e Visualização com Python
Exemplo: Costa Rican Household Poverty Level Prediction
URL (problema): https://www.kaggle.com/c/costa-rican-household-poverty-prediction
URL (solução): https://www.kaggle.com/ivarvb/costa-rican-household-poverty-level-prediction
Autor: Ivar Vargas Belizario
E-mail: ivar@usp.br
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import zipfile

"""
===================================================
I. Ciência de dados
===================================================

===================================================
1. Leitura dos dados para o treino e para o teste
===================================================
"""

zf = zipfile.ZipFile('../../../data/input.zip') 
train = pd.read_csv(zf.open('input/2/train.csv'))
test = pd.read_csv(zf.open('input/2/test.csv'))

#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

"""
===================================================
2. Pre-processamento
===================================================

===================================================
2.1 Limpeza e amostragem
===================================================
"""
# nan para zero 0
#train = train.fillna(0)
#test = test.fillna(0)

# definir as colunas da etiqueta da classe (target) e do identificador (id)
column_target = 'Target'
column_id = 'Id'

# limpar os atributos que apresentam valores nulls
data = train.dropna(axis='columns')
#print (data['agesq'])

# número de instancias antes da amostragem
print ("Total data: ",len(data))

# separação dos atributos: identificador da instancia (id)
# dos atributos data (X) e do atributo que contem a etiqueta da classe (y)
X = data
y = data[column_target]

"""
bins = [0., 2000., 4000., 6000., 8000., 10000.]
labels = [1,2,3,4,5]
X['newfeature1'] = pd.cut(X['agesq'], bins=bins, labels=labels)
X['newfeature2'] = X['SQBmeaned']*X['agesq']
"""

# porcentagem para a amostragem
c_sample = 0.99

if c_sample < 1.0:
    X_null, X, y_null, y = train_test_split(X, y, test_size=c_sample, random_state=0)

ID = X[column_id]
y = X[column_target]
X = X.drop([column_id, column_target], axis=1).select_dtypes(include=[np.number])

train_select_atributes = X.columns

print ("Amostragem: ",len(X))

"""
===================================================
2. Processamento
===================================================

===================================================
2.1 Redução da dimensionalidade (feature selection)
===================================================
"""

"""
model = ExtraTreesClassifier()
model.fit(X, y)
imp = model.feature_importances_
names = []
for i in range(len(imp)):
    r = []
    r.append(i)
    r.append(imp[i])
    names.append(r)

names = sorted(names, key=lambda x: x[1], reverse=True)
fenames = []
columns = list(set(train_select_atributes))
for i in range(len(names)):
    fenames.append(columns[names[i][0]])

train_select_atributes = fenames[:30]
print("redddddddddddd ",train_select_atributes)

X = X[train_select_atributes].values
y = y.values
"""

# convertir para arrays
X = X.values
y = y.values

"""
===================================================
3. Modelo (aprendizagem supervisionado):
===================================================

===================================================
3.1. Treinamento:
===================================================
"""

# definir o modelo para a classificação
model = RandomForestClassifier(random_state=0, n_estimators=500)

# modelo de treinamento com k-fold (10-fold)
kf = StratifiedKFold(n_splits=10)
outcomes = []

# para cada fold
for train_index, test_index in kf.split(X, y):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    model.fit(Xtrain, ytrain)
    expected = ytest
    predictions = model.predict(Xtest)

    accuracy = accuracy_score(ytest, predictions)
    outcomes.append(accuracy)

# imprimir a media da acuracia obtida no treinamento
mean_outcome = np.array(outcomes).mean()

print ("Mean Accuracy:", mean_outcome)

"""
===================================================
3.2. Teste:
===================================================
"""

# selecão de atributos igual ao feito com o conjunto de treino
X_test = test[train_select_atributes]
x_test_id = test[column_id]
predictions = model.predict(X_test)

predictions = pd.DataFrame(predictions, columns = [column_target])

# salvar resultados obtidos do conjunto de dados de teste
result = pd.concat([x_test_id, predictions], axis=1, sort=False)
result.to_csv("result.csv", mode = 'w', index=False)

"""
===================================================
II. Visualização do conjunto de dados (projeções)
===================================================
"""

# normalizar os valores das etiquetas do atributo classe 
isfineClass = False
for i in range(len(y)):
    if y[i]==0:
        isfineClass=True
        break;

if isfineClass==False:
    for i in range(len(y)):
        v = y[i]
        y[i] = v-1

# amostragem para a visualização
c_sample = 0.13

if c_sample < 1.0:
    X_null, X, y_null, y = train_test_split(X, y, test_size=c_sample, random_state=0)
print ("Amostragem para a visualização: ", len(X))

# visualização por projeções t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(6, 5))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

for i in range(len(y)):
    v = y[i]
    plt.plot(X_2d[i, 0], X_2d[i, 1], 'o', color=colors[v], alpha=0.4)
# visualiar a projeção
plt.show()
