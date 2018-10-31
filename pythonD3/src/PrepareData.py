#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# E-mail: ivar@usp.br
import numpy as np
import pandas as pd
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.manifold import TSNE

# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
import zipfile

#from boxPlot import *
class PrepareData():
    """
    Classe
    """
    def __init__(self, pathinput, pathoutput, fn_train, fn_test, c_id, c_class, c_sample=1.0, c_features=[]):
        #return pd.read_csv(filename, header=0, index_col=idcol
        self.pathinput = pathinput
        self.pathoutput = pathoutput

        zf = zipfile.ZipFile(self.pathinput)
        self.train_data = pd.read_csv(zf.open(fn_train), header=0)
        self.test_data = pd.read_csv(zf.open(fn_test), header=0)

        #self.train_data= pd.read_csv(self.path+fn_train, header=0)
        #self.test_data = pd.read_csv(self.path+fn_test, header=0)

        self.c_sample = c_sample

        self.train_id_name = c_id
        self.train_target_name = c_class

        if len(c_features)>0:
            self.train_data = self.train_data[c_features]

        # limpar os atributos que apresentam valores nulls
        self.train_data = self.train_data.dropna(axis='columns')

        self.train_data_target = self.train_data[self.train_target_name].values
        self.train_id = []

        columns = []
        if self.train_id_name != "":
            columns = list(set(self.train_data.columns.values) - set([self.train_id_name, self.train_target_name]))
            self.train_id= self.train_data[self.train_id_name]
            self.test_id = self.test_data[self.train_id_name]
        else:
            columns = list(set(self.train_data.columns.values) - set([self.train_target_name]))
            self.train_id = pd.Series(range(1,len(self.train_data)+1))
            self.test_id = pd.Series(range(1,len(self.test_data)+1))

        self.train_data = self.train_data[columns].select_dtypes(include=[np.number])

        columns = list(set(self.train_data.columns.values) - set([self.train_id_name, self.train_target_name]))
        self.train_data = self.train_data[columns]
        self.test_data = self.test_data[columns]

        #salva todos os nomes dos features
        self.all_features_names = columns

        self.train = self.train_data[self.all_features_names]
        self.test = self.test_data[self.all_features_names]
        self.train_target = self.train_data_target

        #fazer sampling
        #self.make_sample(c_sample)

    def reload(self):
        self.train = self.train_data[self.all_features_names]
        self.test = self.test_data[self.all_features_names]
        self.train_target = self.train_data_target

    def make_sample(self, c_sample):
        if c_sample<1.0:
            self.c_sample = c_sample
            Xnull, self.train, ynull, self.train_target = train_test_split(self.train_data, self.train_data_target, test_size=c_sample, random_state=0)

    def make_normalization(self, type):
        if type == 1:
            self.train = self.make_normalization_column_min_max(self.train_data)
            self.test = self.make_normalization_column_min_max(self.test_data)

        elif type == 2:
            self.train = self.make_normalization_column_standard(self.train_data)
            self.test = self.make_normalization_column_standard(self.test_data)

    def make_normalization_column_min_max(self, data):
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(data)
        return pd.DataFrame(x_scaled)

    def make_normalization_column_standard(self, data):
        min_max_scaler = preprocessing.StandardScaler()
        x_scaled = min_max_scaler.fit_transform(data)
        return pd.DataFrame(x_scaled)

    def make_dim_red(self, type, v_dimred=13):
        if type == 1:
            self.make_dim_red_feature_importance_ExtraTreesClassifier(v_dimred)
        #elif type == 2:
        #    self.make_dim_red_feature_importance_RandomForestClassifier(v_dimred)

    def make_dim_red_feature_importance_ExtraTreesClassifier(self, num_dim=13):
        X = self.get_train().values
        Y = self.get_train_target()
        # feature extraction
        model = ExtraTreesClassifier()
        model.fit(X, Y)
        imp = model.feature_importances_
        names = []
        for i in range(len(imp)):
            r = []
            r.append(i)
            r.append(imp[i])
            names.append(r)

        names = sorted(names, key=lambda x: x[1], reverse=True)
        fenames = []
        columns = list(set(self.train_data.columns.values))
        for i in range(len(names)):
            fenames.append(columns[names[i][0]])

        #print(fenames[:num_dim])
        self.train = self.train_data[fenames[:num_dim]]
        self.test  = self.test_data[fenames[:num_dim]]

    def get_all_feature_names(self):
        return self.all_features_names

    def get_train(self):
        return self.train

    def get_train_target(self):
        return self.train_target

    def get_test(self):
        return self.test

    def get_test_id(self):
        return self.test_id

    def write_data_format(self, fout):
        data = self.get_train()
        labelcol = self.get_train_target()

#        print (labelcol)
#        exit()
        columns = list(set(data.columns.values))
        file = open(fout,"w")
        file.write("DY" + '\n')
        file.write(str(len(data)) + '\n')
        file.write(str(len(columns)) + '\n')
        file.write(';'.join([str(x) for x in columns]) + '\n')
        for index, row in data.iterrows():
#            print(index)
            nrow = row[columns]
            id = index
            if self.train_id_name != "":
                id=row[self.train_id_name]

            v = str(id)+";"+';'.join([str(x) for x in nrow])+";"+str(labelcol[index])
#            v = (str(row[sel_id])+";"+v+";"+str(row[sel_class]))
            file.write(v + '\n')

        file.close()
#        names = self.train.columns.values
#            print (v)

    @staticmethod
    def load_forest_cover_type_prediction():
        PD = PrepareData("../data/input.zip", "../data/output/1/", "input/1/train.csv", "input/1/test.csv", "Id", "Cover_Type")
        return PD

    @staticmethod
    def load_costa_rica():
        PD = PrepareData("../data/input.zip", "../data/output/2/", "input/2/train.csv", "input/2/test.csv", "Id", "Target")
        return PD

    @staticmethod
    def load_digit_recognizer():
        PD = PrepareData("../data/input.zip", "../data/output/3/", "input/3/train.csv", "input/3/test.csv", "", "label")
        return PD
