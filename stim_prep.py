#!/usr/bin/env python
# coding: utf-8
import os
import torch
import numpy as np
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.base import ip
from dnnbrain.dnn.core import Mask
from dnnbrain.dnn.algo import SynthesisImage
from os.path import join as pjoin
from os.path import exists as pexist
import matplotlib.pyplot as plt


init_image = None
unit = (6,6)
reg_meth = 'TV'
lr = 0.1
reg_lambda = 0.5
n_iter = 150
GB_radius=0.2
factor = 0.3
unit_info = {'conv2':[[2],[186]],
            'conv3':[[157],[21]],
            'conv4':[[43],[198]],
            'conv5':[[145],[162]]}


        
class StimPrep:
    """
    Generate optimal image based on net_info
    """
    def __init__(self, dnn, layer=None, channel=None, unit=None,
                 activ_metric='mean', regular_metric=None, precondition_metric=None, smooth_metric=None):
                 
        """
        """
        self.dnn = dnn
        
        
        
        self.dnn.eval()  
        #init synthesisImage
        self.syn = SynthesisImage(self.dnn)
        self.syn.set_metric(activ_metric, regular_metric, precondition_metric, smooth_metric)
        self.syn.set_utiliz(False, True)
        self.syn.set_layer(self.layer, self.channel)
    
    
    def 
    def find_para(self, reg_lambda, factor, top, nrun):
        """
        give a parameter range to find the top parameters in nruns
        
        Parameters
        ----------            
        reg_lambda[ndarray]
        factor[ndarray]
        top[int]
        nrun[int]
        
        Return
        ----------
        top_param[dict]
        """
        if isinstance(reg_lambda, np.ndarray) & isinstance(factor, np.ndarray):
            for l_len in range(reg_lambda.shape):
                for f_len in range(factor.shape):
                    lam = reg_lambda[l_len]
                    fac = factor[f_len]
                    op_img = self.syn.synthesize(None, unit, lr, reg_lambda, 
                                                 n_iter, '.', None, GB_radius, factor, step=50)
        
    def gen_opt(self, top_param, subnum):
        """
        Parameter:
        ----------
        top_param[dict]
        subnum[int]
        
        Returns
        ---------
        optimal[dict]  pickle 
            key:top-sub  value:ndarray
        """
        
        op_img = self.syn.synthesize(init_image, unit, lr, reg_lambda, n_iter, '.', None, GB_radius, factor, step=50)
        #op_img = op_img.transpose(1,2,0)            
        act = synthesis.activ_losses[-1]
        trg = f'lambda-{reg_lambda}_radius-{GB_radius}-factor-{factor}_iter{n_iter}_act-{-act}.jpg'


    def gen_tran(self, optimal, startpoint, axis, length, step):
        """
            initialize a startpoint to translate, generating a list of stimuli
    
        Parameters
        ----------
        optimal[dict]
        startpoint[tuple]
        axis[str]
        length[int]
        step[int]

        Returns
        -------
        opt_tran[dict]
            
        """
        pass
    
    
    def gen_rot():
        """
        

        Parameters
        ----------
        def gen_rot : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass
    
    def gen_sca():
        """


        Parameters
        ----------
        def gen_sca : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass


    def gen_na():
        """
        Generate natural images
        
        Parameters
        ----------
        
        
        Returns
        -------
        None.

        """
        
        
    def extr_act(self, stim, ):
        """
        

        Parameters
        ----------
        stim[packle]

        Returns
        -------
        act[csv]
            row_name:top_subnum  column_name:inv_index  value:dnn.act 

        """
        
    def rear_act(self, act):
        """
        Rearrange the act csv to get specific pic's activation

        Parameters
        ----------
        act[csv]
            row_name:top_subnum  column_name:inv_index  value:dnn.act 
        Returns
        -------
        re_act[csv]

        """
    
    
    
    
    
    for layer in net_info.keys():
        for chn in net_info[layer]:
            save_path = f'/nfs/s2/userhome/zhouming/workingdir/out/Inv_opt/out/conv5_relu/chn{chn[0]-1}/image2/'
            if not pexist(save_path):
                os.mkdir(save_path)
            dnn = AlexNet()
            mask = Mask(layer,chn)
        if not pexist(save_path):
            os.mkdir(save_path)
        img_out = ip.to_pil(op_img,True)
        img_out.save(pjoin(save_path, trg))
