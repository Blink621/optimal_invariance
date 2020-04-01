#!/usr/bin/env python
# coding: utf-8
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
        if np.logical_xor(layer is None, channel is None):
            raise ValueError("layer and channel must be used together!")
        if layer is not None:
            self.set_unit(layer, channel, unit)
        self.dnn = dnn
        self.dnn.eval()
        #init synthesisImage
        self.syn = SynthesisImage(self.dnn)
        self.syn.set_metric(activ_metric, regular_metric, precondition_metric, smooth_metric)
        self.syn.set_utiliz(False, True)
        self.syn.set_layer(self.layer, self.channel)

    def set_unit(self, layer, channel, unit):
        """
        Set layer, channel, unit and its corresponding rf info

        Parameters:
        ----------
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        unit[tuple]: the center unit's location in each channel's feature map
        """
        self.mask = Mask()
        self.mask.set(layer, channels=[channel])    
        self.unit = unit
        #the num means the receptive field size in different layers
        rf_info = {'conv2':51, 'conv3':99, 'conv4':131, 'conv5':163}
        self.rf_size = rf_info[self.layer]

    def rotate(self, img, angle):
        """
        img:[ndarray] 
        """
        img = img.rotate(angle)
        pic = np.array(img).astype('uint8')
        x = np.arange(224)
        y = np.arange(224)
        x,y = np.meshgrid(x,y)
        erea = (x-223/2)**2+(y-223/2)**2
        pic[erea>(self.rf_size/2)**2,:] = [127,127,127]
        return pic

    def find_para(self, reg_lambda, factor, top, nruns):
        """
        give a parameter range to find the top parameters in nruns
        
        Parameters
        ----------            
        reg_lambda[ndarray]
        factor[ndarray]
        top[int]
        nruns[int]
        
        Return
        ----------
        top_param[dict]
        """
        if not isinstance(reg_lambda, np.ndarray) or not isinstance(factor, np.ndarray):
            raise TypeError('Both reg_lambda and factor only support ndarray')
        else:
            #init dataframe for sorting
            para_act = pd.DataFrame(columns=['lambda','factor','act'])
            for l_len in range(reg_lambda.shape):
                for f_len in range(factor.shape):
                    lam = reg_lambda[l_len]
                    fac = factor[f_len]
                    act_all = np.zeros(shape=(nruns))
                    #certify img's stability
                    for run in range(nruns):
                        op_img = self.syn.synthesize(None, self.unit, 0.1, lam, 150,
                                                     '.', None, 0.2, fac, step=50)
                        img = op_img[np.newaxis,:,:,:]
                        act = self.dnn.compute_activation(img, self.mask).get(self.layer)[0,0,self.unit[0],self.unit[1]]
                        act_all[run] = act
                    act_sta = np.mean(act_all)
                    #add values to dataframe
                    info = pd.DataFrame({'lambda':lam,'factor':fac,'act':act_sta}, index=[0])
                    para_act = para_act.append(info, ignore_index=True)
            #sort the dataframe to get top 20 parameters
            para_act = para_act.sort_values(by=['act'], ascending=False).reset_index(drop=True)
            para_act = para_act.loc[:(top+1),:]
            #generate top parameters dict
            top_param = para_act.to_dict(orient='index')
            # store top_para as pickle
            cur_path = os.getcwd()
            store_folder = pjoin(cur_path, 'DataStore')
            if not os.path.exists(store_folder):
                os.mkdirs(store_folder)
            file = pjoin(store_folder, 'TopParameters.pickle')
            with open(file, 'wb') as f:
                pickle.dump(top_param, f)

            return top_param

    def gen_opt(self, top_param, subnum):
        """
        Parameter:
        ----------
        top_param[dict]
        subnum[int]
        
        Returns
        ---------
        optimal[dict]  pickle 
            key:top_sub  value:ndarray
        """
        #init dict
        optimal = dict()
        for top in top_param.keys():
            para = top_param[top]
            lam = para['lambda']
            fac = para['factor']
            for sub in range(subnum):
                op_img = self.syn.synthesize(None, self.unit, 0.1, lam, 150,
                                             '.', None, 0.2, fac, step=50)
                #transpose its shape to (224,224,3)
                op_img = op_img.transpose(1,2,0)    
                #add values to the dict
                op_key = f'top{top+1}_sub{sub+1}'
                optimal[op_key] = op_img
       #save the dict using pickle
        cur_path = os.getcwd()
        store_folder = pjoin(cur_path,'DataStore')
        if not os.path.exists(store_folder):
            os.mkdirs(store_folder)
        file = pjoin(store_folder, 'OptimalImages.pickle')
        with open(file,'wb') as f:
            pickle.dump(optimal,f)
        return optimal

    def gen_tran(self, optimal, axis):
        """
        Generating a list of stimuli based on its unit info
        Only support the half of the original stimuli enter the Left margin of the receptive field
        to move to the symmetrical on the right in stride 1
        Parameters
        ----------
        optimal[dict]
        axis[str]: 'X' or 'Y'

        Returns
        -------
        opt_tran[dict]
            
        """
        if axis != 'X' or axis != 'Y':
            raise ValueError('axis only support X and Y!')
        else:
            opt_tran = dict()
            for org in optimal.keys():
                op_img = optimal[org]
                #crop op_img according to its rf_size
                center = int((224+1)/2)
                span = int((self.rf_size-1)/2)
                op_img = op_img[center-span:center+span, center-span:center+span,:]
                #move the oringal image to generate new stimulus
                center_new = int((448+1)/2)
                start_y = int((448-self.rf_size)/2)
                start_x = 224-self.rf_size 
                #translate in axis X
                if axis == 'X':
                    for move in range(start_x, center_new+1):
                        #init a background waiting for paste, its RGB:(127,127,127)
                        bkg = np.zeros((448,448,3),dtype=np.uint8)
                        bkg[:,:,:] = 127
                        bkg = Image.fromarray(bkg)
                        bkg.paste(op_img, (move, start_y))
                        op_trg = np.asarray(bkg.crop(112,112,336,336))
                        #add vaues to dict
                        tr_key = f'{org}_move:{move-224}'
                        opt_tran[tr_key] = op_trg
                #translate in axis Y
                else:
                    for move in range(start_x, center_new+1):
                        #init a background waiting for paste, its RGB:(127,127,127)
                        bkg = np.zeros((448,448,3),dtype=np.uint8)
                        bkg[:,:,:] = 127
                        bkg = Image.fromarray(bkg)
                        bkg.paste(op_img, (start_y, move))
                        op_trg = np.asarray(bkg.crop(112,112,336,336))
                        #add vaues to dict
                        tr_key = f'{org}_move:{move-224}'
                        opt_tran[tr_key] = op_trg
            #save the dict using pickle
            cur_path = os.getcwd()
            store_folder = pjoin(cur_path, 'DataStore')
            if not os.path.exists(store_folder):
                os.mkdirs(store_folder)
            file = pjoin(store_folder, 'TransferStimuli.pickle')
            with open(file, 'wb') as f:
                pickle.dump(opt_tran, f)
            return opt_tran
    
    def gen_rot(self, optimal, interval):
        """

        Parameters
        ----------
        optimal[dict]
        interval[int] the interval of degrees
        
        Returns
        -------
        opt_rot[dict]

        """
        opt_rot = dict()
        for org in optimal.keys():
            op_img = optimal[org]
            #rotate op_img based on degree
            for degree in range(0, 360, interval):
                op_rog = self.rotate(op_img, degree)
                if degree > 180:
                    degree = degree%180 -180
                ro_key = f'{org}_rot:{degree}'
                opt_rot[ro_key] = op_rog
            #save the dict using pickle
            cur_path = os.getcwd()
            store_folder = pjoin(cur_path, 'DataStore')
            if not os.path.exists(store_folder):
                os.mkdirs(store_folder)
            file = pjoin(store_folder, 'RotateStimuli.pickle')
            with open(file, 'wb') as f:
                pickle.dump(opt_rot, f)

            return opt_rot
    
    def gen_sca(self, optimal, num):
        """


        Parameters
        ----------
        optimal[dict]
        num[int] the total nums of scale stimuli

        Returns
        -------
        None.

        """
        opt_sca = dict()
        for org in optimal.keys():
            op_img = optimal[org]
            #crop op_img according to its rf_size
            center = int((224+1)/2)
            span = int((self.rf_size-1)/2)
            op_img = op_img[center-span:center+span, center-span:center+span,:]
            for per in range(1, num+1):
                #init a background waiting for paste, its RGB:(127,127,127)
                bkg = np.zeros((224,224,3),dtype=np.uint8)
                bkg[:,:,:] = 127
                bkg = Image.fromarray(bkg)
                #scale op_img based on num
                percentage = per/num
                op_rog = transform.rescale(op_img, [percentage,percentage])
                #paste op_rog on bkg!!!!!waiting to fix
                bkg.paste(op_rog,)
                #add values to dict
                sc_key = f'{org}_sca:{per}/{num}'
                opt_sca[sc_key] = op_rog
            #save the dict using pickle
            cur_path = os.getcwd()
            store_folder = pjoin(cur_path, 'DataStore')
            if not os.path.exists(store_folder):
                os.mkdirs(store_folder)
            file = pjoin(store_folder, 'ScaleStimuli.pickle')
            with open(file, 'wb') as f:
                pickle.dump(opt_sca, f)
            return opt_sca

    def gen_na(self):
        """
        Generate natural images
        
        Parameters
        ----------
        
        
        Returns
        -------
        None.

        """

    def extr_act(self, stim, topnum, subnum):
        """
        process one stimuli dict to csv file

        Parameters
        ----------
        stim[pickle]
        topnum [int]: num of parameters
        subnum [int]: num of repeats
        
        Returns
        -------
        act[csv]
            row_name:top_subnum  column_name:inv_index  value:dnn.act 

        """
        # prepare name
        label = list(stim.keys())[0]
        if 'move' in label:
            type_name = 'TI'
        elif 'rot' in label:
            type_name = 'RI'
        elif 'sca' in label:
            type_name = 'SI'
        # prepare dict
        activ_dict = {}
        # extract activation
        for top in range(topnum):
            for sub in range(subnum):
                picname = f'top{top+1}_sub{sub+1}'
                activ_list = []
                cloumn_ist = []
                for key in stim.keys():
                    if picname in key:
                        level = eval(key[key.rfind(':')+1:])
                        stimuli = stim[key]
                        dnn_input = stimuli[n.newaxis,:,:,:].transpose(0,3,1,2)
                        activ = dnn.compute_activation(dnn_input,self.mask).get(self.layer)[0,0,*self.unit]
                        activ_list.append(activ)
                        column_list.append(level)
                activ_dict[picname]=activ_list

        df_activ = pd.DataFrame(activ_dict)
        df_activ = pd.DataFrame(np.array(df_activ).transpose(),
                                index=list(activ_dict.keys()),columns=column_list)
        df_activ = df_activ.sort_index(axis=1)
        # save the df as csv
        cur_path = os.getcwd()
        store_folder = pjoin(cur_path, 'ActData')
        if not os.path.exists(store_folder):
            os.mkdirs(store_folder)
        file_name = pjoin(store_folder,f'{self.layer}_{self.channel}_{type_name}.csv')

        df_activ.to_csv(file_name)

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
        pass
    
