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


   
class StimPrep:
    """
    Generate optimal image based on net_info
    """
    def __init__(self, dnn, activ_metric='mean', regular_metric=None, 
                 precondition_metric=None, smooth_metric=None):
                 
        """
        """        

        self.dnn = dnn
        self.dnn.eval()
        #init synthesisImage
        self.syn = SynthesisImage(self.dnn)
        self.syn.set_metric(activ_metric, regular_metric, precondition_metric, smooth_metric)
        self.syn.set_utiliz(False, True)


    def set_unit(self, layer, channel, unit):
        """
        Set layer, channel, unit and its corresponding rf info

        Parameters:
        ----------
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        unit[tuple]: the center unit's location in each channel's feature map
        """
        if np.logical_xor(layer is None, channel is None):
            raise ValueError("layer and channel must be used together!")
        elif unit is None:
            raise ValueError("unit must be used!")
        self.mask = Mask()
        self.mask.set(layer, channels=[channel])
        self.layer = layer
        self.channel = channel
        self.syn.set_layer(self.layer, self.channel)
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
            para_df = pd.DataFrame(columns=['lambda','factor','act'])
            for l_len in range(reg_lambda.size):
                for f_len in range(factor.size):
                    lam = reg_lambda[l_len]
                    fac = factor[f_len]
                    act_all = np.zeros(shape=(nruns))
                    #certify img's stability
                    for run in range(nruns):
                        op_img = self.syn.synthesize(None, self.unit, 0.1, lam, 10,
                                                     '.', None, 0.2, fac, step=5)
                        op_img = ip.to_pil(op_img,True)
                        op_img = np.array(op_img).transpose(2,0,1)
                        img = op_img[np.newaxis,:,:,:]
                        
                        act = self.dnn.compute_activation(img, self.mask).get(self.layer)[:,0,self.unit[0],self.unit[1]]
                        act_all[run] = act
                    act_sta = np.mean(act_all)
                    #add values to dataframe
                    info = pd.DataFrame({'lambda':lam,'factor':fac,'act':act_sta}, index=[0])
                    para_df = para_df.append(info, ignore_index=True)
            # sort the dataframe to get top parameters
            para_df = para_df.sort_values(by=['act'], ascending=False).reset_index(drop=True)
            para_act = para_df.iloc[0:top,:]
            # generate top parameters dict
            top_param = para_act.to_dict(orient='index')
            # store top_para as pickle
            cur_path = os.getcwd()
            store_folder = pjoin(cur_path, 'DataStore')
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            file = pjoin(store_folder, f'TopParameters-{self.layer}_{self.channel}.pickle')
            with open(file, 'wb') as f:
                pickle.dump(top_param, f)

            return top_param

    def gen_opt(self, top_param, subnum):
        """
        Parameter:
        ----------
        top_param[dict,str]
        subnum[int]
        
        Returns
        ---------
        optimal[dict]  pickle 
            key:top_sub  value:ndarray
        """
        # Type check
        if not (isinstance(top_param,str) or isinstance(top_param,dict)):
            raise TypeError('Only dict or .pickle file path is available')
        # Load dict from pickle
        if isinstance(top_param,str):
            if not os.path.exists(top_param):
                raise NameError('No pickle file exists')
            with open(top_param,'rb') as f:
                top_param = pickle.load(f)
        # init out put dict
        optimal = dict()
        for top in top_param.keys():
            para = top_param[top]
            lam = para['lambda']
            fac = para['factor']
            for sub in range(subnum):
                op_img = self.syn.synthesize(None, self.unit, 0.1, lam, 5,
                                             '.', None, 0.2, fac, step=10)
                # transpose its shape to (224,224,3)
                op_img = op_img.transpose(1,2,0)    
                # add values to the dict
                op_key = f'top{top+1}_sub{sub+1}'
                optimal[op_key] = op_img
        # save the dict using pickle
        cur_path = os.getcwd()
        store_folder = pjoin(cur_path,'DataStore')
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        file = pjoin(store_folder, f'OptimalImages-{self.layer}_{self.channel}.pickle')
        if os.path.exists(file):
            os.remove(file)
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
        # Type check
        if not (isinstance(optimal,str) or isinstance(optimal,dict)):
            raise TypeError('Only dict or .pickle file path is available')
        # Load dict from pickle
        if isinstance(optimal,str):
            if not os.path.exists(optimal):
                raise NameError('No pickle file exists')
            with open(optimal,'rb') as f:
                optimal = pickle.load(f)        
        
        # Run stimuli
        if not axis in ['X','Y'] :
            raise ValueError('axis only support X and Y!')
        else:
            opt_tran = dict()
            for org in optimal.keys():
                op_img = optimal[org]
                #crop op_img according to its rf_size
                center = int((224+1)/2)
                span = int((self.rf_size-1)/2)
                op_img = op_img[center-span:center+span, center-span:center+span,:].astype('uint8')
                #op_img = ip.to_pil(op_img.transpose(2,0,1))
                op_img = Image.fromarray(op_img)
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
                        op_trg = np.asarray(bkg.crop((112,112,336,336)))
                        #add vaues to dict
                        tr_key = f'{org}_move:{move-224+int(self.rf_size/2)}'
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
                os.makedirs(store_folder)
            file = pjoin(store_folder, f'TransferStimuli-{self.layer}_{self.channel}.pickle')
            if os.path.exists(file):
                os.remove(file)
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
        # Type check
        if not (isinstance(optimal,str) or isinstance(optimal,dict)):
            raise TypeError('Only dict or .pickle file path is available')
        # Load dict from pickle
        if isinstance(optimal,str):
            if not os.path.exists(optimal):
                raise NameError('No pickle file exists')
            with open(optimal,'rb') as f:
                optimal = pickle.load(f)        
        
        opt_rot = dict()
        for org in optimal.keys():
            op_img = optimal[org].astype('uint8')
            op_img = ip.to_pil(op_img.transpose(2,0,1))
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
            os.makedirs(store_folder)
        file = pjoin(store_folder, f'RotateStimuli-{self.layer}_{self.channel}.pickle')
        if os.path.exists(file):
            os.remove(file)            
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
        # Type check
        if not (isinstance(optimal,str) or isinstance(optimal,dict)):
            raise TypeError('Only dict or .pickle file path is available')
        # Load dict from pickle
        if isinstance(optimal,str):
            if not os.path.exists(optimal):
                raise NameError('No pickle file exists')
            with open(optimal,'rb') as f:
                optimal = pickle.load(f)
        
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
                os.makedirs(store_folder)
            file = pjoin(store_folder, f'ScaleStimuli-{self.layer}_{self.channel}.pickle')
            if os.path.exists(file):
                os.remove(file)
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
        stim[dict/pickle]
        topnum [int]: num of parameters
        subnum [int]: num of repeats
        
        Returns
        -------
        act[csv]
            row_name:top_subnum  column_name:inv_index  value:dnn.act 

        """
        # Type check
        if not (isinstance(stim,str) or isinstance(stim,dict)):
            raise TypeError('Only dict or .pickle file path is available')
        # Load dict from pickle
        if isinstance(stim,str):
            if not os.path.exists(stim):
                raise NameError('No pickle file exists')
            with open(stim,'rb') as f:
                stim = pickle.load(f)        
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
                column_list = []
                stimuli_set = np.random.rand(224,224,3).astype('uint8')[np.newaxis,:,:,:].transpose(0,3,1,2)
                for key in stim.keys():
                    if picname in key:
                        level = eval(key[key.rfind(':')+1:])
                        stimuli = stim[key]
                        dnn_input = stimuli[np.newaxis,:,:,:].transpose(0,3,1,2)
                        stimuli_set = np.concatenate((stimuli_set,dnn_input),axis=0)
                        column_list.append(level)
                
                stimuli_set = np.delete(stimuli_set,0,axis=0)
                activ = dnn.compute_activation(stimuli_set,self.mask).get(self.layer)[:,0,self.unit[0],self.unit[1]]
                activ_list = list(activ)
                activ_dict[picname]=activ_list

        df_activ = pd.DataFrame(activ_dict)
        df_activ = pd.DataFrame(np.array(df_activ).transpose(),
                                index=list(activ_dict.keys()),columns=column_list)
        df_activ = df_activ.sort_index(axis=1)
        # save the df as csv
        cur_path = os.getcwd()
        store_folder = pjoin(cur_path, 'ActData')
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        file_name = pjoin(store_folder,f'{self.layer}_{self.channel}_{type_name}.csv')
        df_activ.to_csv(file_name)

    


init_image = None
unit ={'conv2': (13,13),'conv3':(6,6)}
reg_meth = 'TV'
topnum = 2
subnum = 2
lr = 0.1
reg_lambda = np.array([0.001])
n_iter = 150
factor = np.arange(0.2,0.5,0.05)
unit_info = { 'conv2':[2,186],
              'conv3':[157,21],
              'conv4':[43,198],
              'conv5':[145,162]}
dnn = AlexNet()
experiment = StimPrep(dnn)
experiment.set_unit('conv3',21,(6,6))
pth = r'/nfs/s2/userhome/zhouming/workingdir/optimal_invariance/DataStore'
file = pjoin(pth,'TransferStimuli-conv3_21.pickle')
experiment.extr_act(file,2,2)


for layer,chn in unit_info.items():
    for i in range(len(chn)):
        experiment.set_unit(layer,chn[i],unit[layer])
        para_dict = experiment.find_para(reg_lambda,factor,top=topnum,nruns=2)
        image_set = experiment.gen_opt(para_dict,subnum)
        transfer_stimuli = experiment.gen_tran(image_set,'X')
        rotate_stimuli = experiment.gen_rot(image_set,30)
    for stimuli in [transfer_stimuli,rotate_stimuli]:
        experiment.extr_act(stimuli,topnum,subnum)



