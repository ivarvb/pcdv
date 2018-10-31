#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# E-mail: ivar@usp.br

import tornado.ioloop
import tornado.web
import tornado.httpserver
from multiprocessing import cpu_count

#from sklearn import datasets
from Visualization import *
from Classification import *
from PrepareData import *
from flask import json

import glob
import sys

HOST = 'localhost'
PORT = 9999

PD_01 = PrepareData.load_forest_cover_type_prediction()
PD_02 = PrepareData.load_costa_rica()
PD_03 = PrepareData.load_digit_recognizer()


class Index(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", HOST=HOST, PORT=PORT)
        #self.write(myString)


class PipeLine(tornado.web.RequestHandler):
    def set_default_headers(self):
        #print "setting headers!!!"
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def select_data(self, v_data):
        data = []
        if v_data == 1:
            data = PD_01
        elif v_data == 2:
            data = PD_02
        elif v_data == 3:
            data = PD_03

        return data

    def get_pipeline_vis(self):
        v_data = int(self.get_argument('data'))
        v_sample = float(self.get_argument('samples'))
        #v_dimred = int(self.get_argument('dimered'))
        #v_datanor = int(self.get_argument('datanor'))
        v_visuali = int(self.get_argument('visuali'))
        v_classif = int(self.get_argument('classif'))

        #select dataset
        data = self.select_data(v_data)

        #reload data
        data.reload()

        #make normalization
        #data.make_normalization(v_datanor)

        #make sample
        data.make_sample(v_sample)

        #make dimensionality reduction
        #data.make_dim_red(v_dimred)

        #make visualization
        vz = Visualization(data, v_visuali)
        result = vz.execute()

        self.write(result)

    def get_feature_names(self):
        v_data = int(self.get_argument('data'))
        data = self.select_data(v_data)
        fn = data.get_all_feature_names()
        #fn = json.dumps(fn)

        result = "<select id=\"id_classtarget\" style=\"width:100%\">"
        for i in fn:
            result+="<option value=\""+str(i)+"\">"+str(i)+"</option>"
        result += "</select>"
        self.write(result)

    def get_classification(self):
        v_data = int(self.get_argument('data'))
        v_sample = float(self.get_argument('samples'))
        #v_dimred = int(self.get_argument('dimered'))
        #v_datanor = int(self.get_argument('datanor'))
        v_visuali = int(self.get_argument('visuali'))
        v_classif = int(self.get_argument('classif'))

        #select dataset
        data = self.select_data(v_data)

        #reload data
        data.reload()

        #make normalization
        #data.make_normalization(v_datanor)

        #make sample
        data.make_sample(v_sample)

        #make dimensionality reduction
        #data.make_dim_red(v_dimred)

        cls = Classification(data, v_classif)
        result = cls.execute()

        self.write(result)

    #Get RequestHandler
    def get(self):
        v_tq = int(self.get_argument('tq'))

        if v_tq == 1:
            self.get_pipeline_vis()
        elif v_tq == 2:
            self.get_feature_names()
        elif v_tq == 3:
            self.get_classification()


    #Post RequestHandler
    def post(self):
        username = self.get_argument('username')
        designation = self.get_argument('designation')

        self.write("Wow " + username + " you're a " + designation)


app = tornado.web.Application([
    #init aplications
    (r"/", Index),
    (r"/pl/", PipeLine),
    #load files
    #(r"/src/(.*)",tornado.web.StaticFileHandler, {"path": "./src"},),
    (r"/data/(.*)",tornado.web.StaticFileHandler, {"path": "./data"},),
    (r"/lib/(.*)",tornado.web.StaticFileHandler, {"path": "./lib"},),
])
#app.listen(PORT)

if __name__ == "__main__":

    print ('The server is ready: http://'+HOST+':'+str(PORT)+'/')
    server = tornado.httpserver.HTTPServer(app)
    server.bind(PORT)
    #specify number of subprocess
    #server.start(4)
    server.start(cpu_count())
    tornado.ioloop.IOLoop.current().start()
    #tornado.ioloop.IOLoop.instance().start()




#digits = datasets.load_digits()
#X = digits.data[:1100]
#Y = digits.target[:1100]
#tsne = TSNE(n_components=2, random_state=0)
#X_2d = tsne.fit_transform(X)
