#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


import glob
import sys
import itertools
import json

#from boxPlot import *

class Classification():
    def __init__(self, data, model):
        self.data = data

        self.model = RandomForestClassifier(random_state=0)
        #self.model = KNeighborsClassifier(n_neighbors = 1)

    def acuracy(self, cm):
        accrest = [0.0 for x in range(len(cm[0]))]
        for i in range(0, len(cm[0])):
            accrest[i] = cm[i][i]

        for i in range(0, len(cm[0])):
            sumr = 0.0
            for j in range(0, len(cm[0])):
                sumr+=cm[i][j]
            accrest[i]/=sumr
        return accrest

    def execute_train(self):
        X = self.data.get_train()
        y = self.data.get_train_target()
        #print ("x", "y", len(X), len(y))
        #verificar valores da classe
        isfineClass = False
        for i in y:
            if i==0:
                isfineClass=True
                break;
        if isfineClass==False:
            for i in range(len(y)):
                y[i] = y[i]-1


        #kf = KFold(n_splits=5)
        kf = StratifiedKFold(y, n_folds=10)

        outcomes = []
        accarr = []
        #for train_index, test_index in kf.split(X):
        for train_index, test_index in kf:
            Xtrain, Xtest = X.values[train_index], X.values[test_index]
            ytrain, ytest = y[train_index], y[test_index]
            self.model.fit(Xtrain, ytrain)
            expected = ytest
            predictions = self.model.predict(Xtest)

            accuracy = accuracy_score(ytest, predictions)
            outcomes.append(accuracy)

            cm = metrics.confusion_matrix(expected, predictions)
            acc = self.acuracy(cm)
            accarr.append(acc)

        outcomes = np.array(outcomes)
        mean_outcome = outcomes.mean()

        accarr_means = [0.0 for i in range(len(accarr[0]))]
        for i in range (len(accarr)):
            for j in range (len(accarr[0])):
                accarr_means[j] = accarr[i][j]

        print(accarr_means)

        result = "Mean accuracy: "+format(mean_outcome)+"<br>"
        #print(result)

        #result+= '\t\t'.join([str(x) for x in range (len(accarr_means))])+"<br>"
        #result+= '\t\t'.join([str(x) for x in accarr_means])+"<br>"
        return result

    def execute_test(self):
        X = self.data.get_test()
        x_id = self.data.get_test_id()
        predictions = self.model.predict(X)

        predictions = pd.DataFrame(predictions, columns = ["Label"])
        result = pd.concat([x_id, predictions], axis=1, sort=False)
        result.to_csv(self.data.pathoutput+"result.csv", mode = 'w', index=False)

        result = "<br>results saved (testing):"+self.data.pathoutput+"result.csv"
        #result = ""
        return (result)

    def execute(self):
        trainrest = self.execute_train()
        testrest = self.execute_test()
        #print (testrest)
        return (trainrest+testrest)
