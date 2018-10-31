#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# E-mail: ivar@usp.br

import pandas as pd

from PrepareData import *
class Pipeline():
    """
    Classe

    """
    def __init__(self, filename, c_id, c_class):
        #return pd.read_csv(filename, header=0, index_col=idcol)
        self.data = pd.read_csv(filename, header=0)
        self.c_id = c_id
        self.c_class = c_class

    def execute(self, v_data, v_id, v_class):
        traind = v_data+"/train.csv"
        pd = PrepareData(v_data, v_id, v_class)
        PD = PrepareData("data/1/train.csv", "Id", "Cover_Type", 0.1)

        data = pd.get_data()
        cl = Classification(data)
        res = cl.execute()
        return res

if __name__ == "__main__":
    pip = PrepareData.load_ds_01()
    pip.print_data()
    del pip
