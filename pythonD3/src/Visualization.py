#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# E-mail: ivar@usp.br


from sklearn import datasets
from sklearn.manifold import TSNE
from PrepareData import *
from sklearn.datasets import load_digits

#from boxPlot import *
class Visualization():
    """
    Classe
    """
    def __init__(self, pdata, type):
        #return pd.read_csv(filename, header=0, index_col=idcol)
        self.pdata = pdata
        self.type = type

    def execute(self):
        X_2d = []
        Y = []

        if self.type == 1:
            X_2d, Y = self.tsne()
        #elif self.type == 2:
        #    X_2d, Y = self.plmp()
        #elif self.type == 3:
        #    X_2d, Y = self.lamp()
        #elif self.type == 4:
        #    X_2d, Y = self.lamp()
        #elif self.type == 5:
        #    X_2d, Y = self.umap()

        result = ""
        result = "id\t\tx\t\ty\t\tc\n"
        for i in range(len(X_2d)):
            result+=str(i)+"\t\t"+str(X_2d[i][0])+"\t\t"+str(X_2d[i][1])+"\t\t"+str(Y[i])+"\n"

        return result

    def tsne(self):
        PD = self.pdata
        X = PD.get_train()
        Y = PD.get_train_target()
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(X)

        return X_2d, Y
