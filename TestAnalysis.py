#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:36:59 2020

@author: zhouming
"""
import pickle
import pandas as pd
import numpy as np
from os.path import join as pjoin
import matplotlib.pyplot as plt

# data path
DSpath = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore'
ADpath = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/ActData'

# unit_info
unit_info = { 'conv2':[186]}#,
              #'conv3':[21,157],#2,
              #'conv4':[43,198],#,,
              #'conv5':[145,162]}

# activation compare
TopAct = {}
ExtAct = {}
# top sub parameters
topnum = 5
subnum = 5
# comparation
for layer, chn in unit_info.items():
    for i in range(len(chn)):
        # topact preparation 
        act0 = []
        pklfile = pjoin(DSpath,f'TopParameters-{layer}_{chn[i]}.pickle')
        with open(pklfile, 'rb') as f:
            searchAct = pickle.load(f)
        for j in range(len(searchAct.keys())):
            act0.append(searchAct[j]['act'])
        TopAct[f'{layer}_{chn[i]}'] = act0
        
        # extact preparation
        act1 = []
        file = pjoin(ADpath,f'{layer}_{chn[i]}_TI.csv')
        df = pd.read_csv(file)
        df_temp = df.loc[:,['0']]
        #print(df_temp)
        for j in range(topnum):
            extrAct = np.mean(np.array(df_temp.iloc[subnum*j:subnum*(j+1),:]))
            #print(df_temp.iloc[,0].mean())
            act1.append(extrAct) # 120 column is 0 degree
        print(f'>>>>>>>{layer}_{chn[i]}<<<<<<<<')
        print('activ in searching:',act0)
        print('activ in extractio:',act1)
        plt.scatter(act0,act1,label=f'{layer}_{chn[i]}')
plt.plot([-100,700],[-100,700],lw=3.0)
plt.xlim([-125,800])
plt.ylim([-125,800])
plt.axis('equal')
plt.legend()
plt.show()        

# In[]

import pandas as pd

# =============================================================================
# ac1 = pd.read_csv('/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/ActData/conv2_186_TI.csv')
# ac0 = pd.read_csv('/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/conv2_186_TI.csv')
# ac = np.array(ac1.iloc[:,1:])-np.array(ac0.iloc[:,1:])
# =============================================================================

def img_compare(unit,dtype,pos):
    sfile = f'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore/OptimalImages-{unit}.pickle'
    with open(sfile,'rb') as f:
        stand = pickle.load(f)
    keys = f'top{pos[0]}_sub{pos[1]}'
    stand_pic = stand[keys]

    if dtype == 'TI':
        tfile = f'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore/TransferStimuli-{unit}.pickle'
        keys = f'top{pos[0]}_sub{pos[1]}_move:0'
    elif dtype == 'RI':
        tfile = f'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore/RotateStimuli-{unit}.pickle'
        keys = f'top{pos[0]}_sub{pos[1]}_rot:0'
    
    with open(tfile,'rb') as f:
        test = pickle.load(f)
    test_pic = test[keys]
    plt.imshow(stand_pic-test_pic)
    
    
img_compare('conv2_186','TI',(1,1))
#stand_pic = 
#test_pic = 