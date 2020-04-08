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

class DataStore:
    def __init__(self):
        self.name = 'processed data for all invariance of units'
        self.data = dict()
        
class DataAnalysis:
    """
    Analysis stim's invariance according to the corresponding act csv
    """
    def __init__(self, folder,layer,channel):

        self.path = folder
        self.layer = layer
        self.channel = channel
        self.data = None
        self.datadict = dict()

    def save(self, stype='optimal'):
        '''
        save updated  
        '''
        if not stype in ['optimal','natural'] :
            raise ValueError('stype only support optimal and natural!')
        # check data folder
        path = os.getcwd()
        #print('1')
        if stype == 'optimal':
            datapath = pjoin(path,'AnalysisData')
        else:
            datapath = pjoin(path,'NaAnalysisData')
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        #print('2')
        # read existed file
        datafile = pjoin(datapath, 'W_data.pickle')
        if os.path.exists(datafile):
            origin_file = datafile
            with open(origin_file,'rb') as f:
                old_dict = pickle.load(f)
                old_dict.update(self.datadict)
            os.remove(datafile)
        # store updated pickle
            with open(datafile, 'wb') as f:
                pickle.dump(old_dict, f)
            #print('4')
        else:
            with open(datafile, 'wb') as f:
                pickle.dump(self.datadict, f)
            #print('3')
            

    def gen_mat(self, df_0, df,method):
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
        if method=='meansquare':
            s_act = np.sqrt((array_df.T).dot(array_df).diagonal()) 
        elif method == 'mean':
            s_act = np.mean(array_df,axis=0)
        
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

    def inv_mat(self, dtype, stype='optimal'):
        """
        Plot invariance curve

        Parameters
        ----------
        type[]

        Returns
        -------
        dict[dict]

        """
        if not stype in ['optimal','natural'] :
            raise ValueError('stype only support optimal and natural!')
        if isinstance(type,str):
            dtype = [dtype]
        unit_dict = dict()
        for i in range(len(dtype)):
            filename = f'{self.layer}_{self.channel}_{dtype[i]}.csv'
            file_path = pjoin(self.path, filename)
            self.data = pd.read_csv(file_path)

            temp_data = self.data.iloc[:,1:]
            center = round(len(temp_data.columns)/2)
            temp_d0 = temp_data.iloc[:,center]
            data_mat = self.gen_mat(temp_d0,temp_data,'mean')
            if dtype[i] == 'TI':
                temp_compute = temp_data.iloc[:,int(0.5*center):int(1.5*center)+1]
                data_index = self.compute_index(temp_compute)
                #print(data_index)
            elif dtype[i] == 'RI':
                data_index = self.compute_index(temp_data)
            if stype == 'natural':
                key_name = 'na_' + filename.replace('.csv','_plot')
                index_name = 'na_' + filename.replace('.csv','')
            else:
                key_name = filename.replace('.csv','_plot')
                index_name = filename.replace('.csv','')
            unit_dict[key_name] = data_mat
            unit_dict[index_name] = data_index

        unit_name = f'{self.layer}_{self.channel}'
        self.datadict[unit_name] = unit_dict

        return unit_dict
        
    def plot_tun(self, file, unit, dtype, stype='optimal'):
        """
        Plot tunning curve

        Parameters
        ----------
        file[str] pickle name for dataset
        unit[str] unit name
        dtype[str] data type
        stype[str] optimal or natural
        
        Returns
        -------
        
        """
        if not stype in ['optimal','natural'] :
            raise ValueError('stype only support optimal and natural!')
        if stype == 'optimal':
            datafile = f'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/AnalysisData/{file}'
        else:
            datafile = f'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/NaAnalysisData/{file}'
        with open(datafile, 'rb') as f:
            datastore = pickle.load(f)
        if isinstance(datastore,DataStore):
            datadict = datastore.data
        elif isinstance(datastore,dict):
            datadict = datastore # [dict] keys: 'layer_unit'
        if isinstance(dtype,str):
            dtype = [dtype]
        datafolder = pjoin(os.getcwd(),'Figures')
        if not os.path.exists(datafolder):
            os.makedirs(datafolder)
        for i in range(len(dtype)):
            # prepare plot matrix & index
            if stype == 'optimal':
                indexname = f'{unit}_{dtype[i]}'
            else:
                indexname = f'na_{unit}_{dtype[i]}'
            dataname = f'{indexname}_plot'
            unit_matrix = datadict[unit][dataname]
            unit_index = datadict[unit][indexname]
            # figure plot
            plt.figure(figsize=(8,8))
            plt.plot(unit_matrix[:,0],unit_matrix[:,2],color='g',label='Activ',lw=2.8)
            plt.plot(unit_matrix[:,0],unit_matrix[:,1],color='r',label='r',lw=2.8)
            plt.text(unit_matrix[1,0],0.87,unit,fontsize=18)
            plt.text(unit_matrix[int(4*len(unit_matrix)/10),0],-0.27,f'{dtype[i]}={round(unit_index*100)/100}',fontsize=20)
            if dtype[i]=='TI':
                plt.xticks([unit_matrix[0,0],unit_matrix[int(len(unit_matrix)/2),0],unit_matrix[-1,0]],[-0.5,0,0.5])
                center = round(len(unit_matrix)/2)
                plt.plot([unit_matrix[int(0.5*center),0],unit_matrix[int(1.5*center),0]],[-0.3,-0.3],lw=4.0,color='k')
                plt.plot([unit_matrix[int(0.5*center),0],unit_matrix[int(0.5*center),0]],[0.9,-0.3],ls='--',lw=1.0,color='k')
                plt.plot([unit_matrix[int(1.5*center),0],unit_matrix[int(1.5*center),0]],[0.9,-0.3],ls='--',lw=1.0,color='k')
                plt.xlabel('Relative Distance /rf size')
            if dtype[i]=='RI':
                plt.xlabel('Rotation /degree')
            plt.legend()
            picname = f'{indexname}.png'
            plt.savefig(pjoin(datafolder,picname))
                        
        
        
        
        
        

# In[]
folder = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/ActData'
unit_info = { 'conv5':[145,162],
              'conv4':[43,198],#
              'conv3':[157,21],#,,
              'conv2':[2,186]}
#unit_info = {'conv2':[186]}


for layer,chn in unit_info.items():
    for i in range(len(chn)):
       # print(layer,chn[i])
        analysis = DataAnalysis(folder,layer,chn[i])
        matrix = analysis.inv_mat(['TI','RI'])
        analysis.save(stype='optimal')

for layer,chn in unit_info.items():
    for i in range(len(chn)):
        analysis.plot_tun('W_data.pickle',f'{layer}_{chn[i]}',['TI','RI'])
        
        
file = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/NaAnalysisData/W_data.pickle'
with open(file,'rb') as f:
    dict1 = pickle.load(f)
    print(dict1.keys())


# In[]

Data = DataStore()
folder = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/NaActData'
unit_info = { 'conv5':[145,162],
              'conv4':[43,198],}
              'conv3':[157,21],#,,
              'conv2':[2,186]}
pklfile = pjoin(os.getcwd(),'NaAnalysisData','Whole_data.pickle')
for layer,chn in unit_info.items():
    for i in range(len(chn)):
        print(layer,chn[i])
        analysis = DataAnalysis(folder,layer,chn[i])
        matrix = analysis.inv_mat(['TI','RI'],stype='natural')
        name = f'{layer}_{chn[i]}'
        Data.data[name] = matrix

if os.path.exists(pklfile):
    with open(pklfile,'rb') as f:
        old_data =  pickle.load(f)
        old_data.data.update(Data.data)

    with open(pklfile,'wb') as f:
        pickle.dump(old_data,f)
else: 
    with open(pklfile,'wb') as f:
        pickle.dump(Data,f)

# In[]
unit_info = { 'conv5':[145,162],
              'conv4':[43,198],}
              #'conv3':[157,21],#,,
              #'conv2':[2,186]}
for layer,chn in unit_info.items():
    for i in range(len(chn)):
        analysis.plot_tun('Whole_data.pickle',f'{layer}_{chn[i]}',['TI','RI'],stype='natural')

print('==============OK!============')
# In[]

file = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/AnalysisData/W_data.pickle'
with open(file,'rb') as f:
    dictm = pickle.load(f)
    print(dictm.keys())
TI = []
RI = []
ticks = []
unit_info = { 'conv2':[2,186],
              'conv3':[21,157],#
              'conv4':[43,198],#,,
              'conv5':[145,162]}

for layer,chn in unit_info.items():
    for i in range(len(chn)):
        ticks.append(f'{layer}_{chn[i]}')
        TI.append(dictm[f'{layer}_{chn[i]}'][f'{layer}_{chn[i]}_TI'])
        RI.append(dictm[f'{layer}_{chn[i]}'][f'{layer}_{chn[i]}_RI'])
plt.figure(figsize=(8,8))
plt.plot(np.arange(len(TI)),np.array(TI),color='r',lw=2.5,label='TI',marker='o',markersize=9.0)
plt.plot(np.arange(len(RI)),np.array(RI),color='g',lw=2.5,label='RI',marker='o',markersize=9.0)
plt.xticks(np.arange(len(RI)),ticks)
plt.legend()
plt.savefig(r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/Figures/Layer_index.png')

# In[]
path = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore'
spath =  r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/Figures'
unit_info = { 'conv2':[2,186],
              'conv3':[21,157],#
              'conv4':[43,198],#,,
              'conv5':[145,162]}
act = []
ticks = []
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_position([0.1,0.1,0.6,0.8])
for layer,chn in unit_info.items():
    for i in range(len(chn)):
        actlist=[]
        file = pjoin(path,f'TopParameters-{layer}_{chn[i]}.pickle')
        with open(file,'rb') as f:
            dictpara = pickle.load(f)
        ticks.append(f'{layer}_{chn[i]}')
        for j in range(len(dictpara.keys())):
            actlist.append(dictpara[j]['act'])
        ax.plot(np.arange(len(dictpara.keys())),np.array(actlist),lw=3.0,label=f'{layer}_{chn[i]}')
        
        actmean = np.mean(np.array(actlist))
        
        #plt.plot([0,7],[actmean,actmean],'')
        
plt.title('top parameters mean activation')
leg = ax.legend(loc=2,bbox_to_anchor=(1.05,1.0),borderaxespad=0.)
fig.savefig(f'{spath}/top_parameters.png')

# In[]
apath = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/ActData'
ipath =  r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/AnalysisData/W_data.pickle'
unit_info = { 'conv2':[2,186],
              'conv3':[21,157],#
              'conv4':[43,198],#,,
              'conv5':[145,162]}

with open(ipath,'rb') as f:
    dictm = pickle.load(f)
    print(dictm.keys())
TI = []
RI = []
actmean = []
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_position([0.1,0.1,0.7,0.7])
for layer,chn in unit_info.items():
    for i in range(len(chn)):
        tif = pd.read_csv(pjoin(apath,f'{layer}_{chn[i]}_TI.csv'))
        rif = pd.read_csv(pjoin(apath,f'{layer}_{chn[i]}_TI.csv'))
        actmean.append(np.mean(np.array(tif['0'])))
        
        TI.append(dictm[f'{layer}_{chn[i]}'][f'{layer}_{chn[i]}_TI'])
        RI.append(dictm[f'{layer}_{chn[i]}'][f'{layer}_{chn[i]}_RI'])
        x = [np.mean(np.array(tif['0'])),np.mean(np.array(rif['0']))]
        y = [dictm[f'{layer}_{chn[i]}'][f'{layer}_{chn[i]}_TI'],dictm[f'{layer}_{chn[i]}'][f'{layer}_{chn[i]}_RI']]
        ax.scatter(x[0],y[0],label=f'{layer}_{chn[i]}_TI',marker='D',s=140)
        #plt.scatter(x[1],y[1],label=f'{layer}_{chn[i]}_RI',marker='D')
plt.xlabel('Activation',fontsize=20)
plt.ylabel('Index',fontsize=20)
ax.legend(loc=2,bbox_to_anchor=(1.02,1.0),borderaxespad=0.)
plt.savefig(pjoin(os.getcwd(),'Figures/Activtion&TIndex.png'))
