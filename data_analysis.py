# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:36:57 2020

@author: lenovo
"""
import os
import pickle
import numpy as np
import pandas as pd
from os.path import join as pjoin
import matplotlib.pyplot as plt

class DataAnalysis:
    """
    Analysis stim's invariance according to the corresponding act csv
    """
    def __init__(self, folder,channel,unit):

        self.path = folder
        self.channel = None
        self.unit = None
        self.data = None
        self.datadict = {}

    def save(self):
        '''
        save updated  
        '''
        # check data folder
        path = self.path[0:self.path.rfind('\\',0,len(self.path))]
        datapath = pjoin(path,'data')
        if not os.path.exists(datapath):
            os.makedirs(datapath)

        # read existed file
        if len(os.listdir(datapath)) > 0 :
            origin_file = pjoin(datapatn,os.listdir(datapath)[0])
            with open(origin_file,'rb') as f:
                old_dict = pickle.load(f)
            self.datadict = old_dict.update(self.datadict)
        datafile = pjoin(datapath, 'Whole_data.pickle')

        # store updated pickle
        with open(datafile, 'wb') as f:
            pickle.dump(self.datadict, f)

    def gen_mat(self, df_0, df):
        '''
        df [dataframe] 
        
        return:
        r_act_matrix [ndarray]: pos :: r :: activ
        '''
        # initial plot matrix
        num = len(df.columns)
        r_act_matrix = np.zeros((num, 4))
        # compute normalized activation
        array_df = np.array(df)
        s_act = np.sqrt((array_df.T).dot(array_df).diagonal())
        r_act_matrix[:, 3] = s_act  # last column is original activation
        norm_act = (s_act - np.min(s_act)) / (np.max(s_act) - np.min(s_act))
        r_act_matrix[:, 2] = norm_act  # third column is normalized activation
        # pixel from standard
        r_act_matrix[:, 0] = np.arange(num)  # first colunm is pixel position
        # r value
        for ii in range(num):
            r = df_0.corr(df.iloc[:, ii])
            r_act_matrix[ii, 1] = r  # second column is correlation
        return r_act_matrix

    def compute_index(self,df):
        df_act = df
        cov_mat = np.array(df_act.cov())

        std_diag = np.sqrt(cov_mat.diagonal().reshape(len(cov_mat), 1))
        Std_sum = 0.5 * (np.sum(std_diag.dot(std_diag.T)) - np.sum(cov_mat * np.eye(len(cov_mat))))
        Cov_sum = 0.5 * (np.sum(cov_mat) - np.sum(cov_mat * np.eye(len(cov_mat))))

        Index = Cov_sum / Std_sum

        return Index

    def inv_mat(self, type):
        """
        Plot invariance curve

        Parameters
        ----------
        type[]

        Returns
        -------
        dict[dict]

        """
        if isinstance(type,str):
            type = [type]
        for i in range(len(type)):
            filename = f'{self.channel}_{self.unit}_{type[i]}.csv'
            file_path = pjoin(self.folder, dir, filename)
            self.data = pd.read_csv(file_path)

            temp_data = self.data.iloc[:,1:]
            temp_d0 = temp_data.iloc[:,round(len(temp_data.columns)/2)-1]
            data_mat = gen_mat(temp_d0,temp_data)
            data_index = compute_index(temp_data)

            key_name = filename.replace('.csv','_plot')
            index_name = filename.replace('.csv','')
            unit_dict[key_name] = data_mat
            unit_dcit[index_name] = data_index

        unit_name = f'{self.channel}_{self.unit}'
        self.datadict[unit_name] = unit_dict

        return unit_dict
        
    def plot_tun(self, act):
        """
        Plot tunning curve

        Parameters
        ----------
        act[csv]

        Returns
        -------
        
        """
