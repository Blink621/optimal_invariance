#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:22:03 2020

@author: zhouming
"""
import torch,pickle,os
from os.path import join as pjoin
from PIL import Image
import numpy as np
import pandas as pd
from skimage import transform
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.base import ip
from dnnbrain.dnn.core import Mask
from dnnbrain.dnn.algo import SynthesisImage
import matplotlib.pyplot as plt


pth = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore'
file = pjoin(pth,'TransferStimuli-conv3_21.pickle')
with open(file,'rb') as f:
    img_set = pickle.load(f)
activ_dict = {}
mask = Mask(layer='conv3',channels=[21])
pic = img_set['top2_sub2_move:50'].astype('uint8')
plt.subplot(121)
plt.imshow(pic)
pic = pic[np.newaxis,:,:,:].transpose(0,3,1,2)
activ = dnn.compute_activation(pic,mask).get('conv3')[0,0,6,6]
activ_dict['top2_sub2_move:50'] = activ
pic = img_set['top2_sub2_move:-49'].astype('uint8')
plt.subplot(122)
plt.imshow(pic)
pic = pic[np.newaxis,:,:,:].transpose(0,3,1,2)
activ = dnn.compute_activation(pic,mask).get('conv3')[0,0,6,6]
activ_dict['top2_sub2_move:-49'] = activ
print(activ_dict)

'''
pth = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore'
file = pjoin(pth,'TransferStimuli-conv3_21.pickle')
name = ['top2_sub2_move:50','top2_sub2_move:-49']
activ_dict = {}
for key in img_set.keys():
    i = 1
    if key in name:
        name_list.append(key)
        stimuli = img_set[key].astype('uint8')
        plt.subplot(1,2,i)
        plt.imshow(stimuli)
        i += 1
        dnn_input = stimuli[np.newaxis,:,:,:].transpose(0,3,1,2)
        activ = dnn.compute_activation(dnn_input,mask).get('conv3')[0,0,6,6]
        activ_dict[key]=activ
print(activ_dict)
'''

'''
pth = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore'
pdir = os.listdir(pth)

for file in pdir:
    if 'Transfer' in file:
        filename = file
        #layer = filename[filename.rfind('-')+1:filename.rfind('_')]
       # chn = int(filename[filename.rfind('_')+1:filename.rfind('.')])
        #experiment.set_unit(layer,chn,unit)
        with open(pjoin(pth,file),'rb') as f:
            img_set = pickle.load(f)
        #experiment.gen_rot(img_set,30)        
        print(img_set.keys())
for file in pdir:
    if 'RotateStimuli' in file:
        filename = file
        layer = filename[filename.rfind('-')+1:filename.rfind('_')]
        chn = int(filename[filename.rfind('_')+1:filename.rfind('.')])
        experiment.set_unit(layer,chn,unit)
        with open(pjoin(pth,file),'rb') as f:
            stimuli_set = pickle.load(f)
       # print(f'=========={filename}===========',stimuli_set.keys())
        experiment.extr_act(stimuli_set,topnum,subnum)
'''
print('==============!!OK!!====================')