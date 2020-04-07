import pickle,os,copy,re
from os.path import join as pjoin
from PIL import Image
import numpy as np
import pandas as pd
from skimage import transform
from dnnbrain.dnn.base import ip
from dnnbrain.dnn.core import Mask
from dnnbrain.dnn.algo import SynthesisImage
import matplotlib.pyplot as plt

class StimPrep:
    """
    Generate optimal image based on net_info
    """
    def __init__(self, dnn, activ_metric='mean', regular_metric='TV', 
                 precondition_metric=None, smooth_metric='Fourier'):
                 
        """
        Parameters:
        ----------
        dnn[DNN]: dnnbrain's DNN object
        metric[str] : four metrics used in synthesisImage
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
        #the first num means the receptive field size, the second num means the feature map size
        rf_info = {'conv2':[51,27], 'conv3':[99,13], 'conv4':[131,13], 'conv5':[163,13]}
        self.rf_size = rf_info[self.layer][0]
        self.fm_size = rf_info[self.layer][1]

    def rotate(self, img, angle):
        """
        rotate function used in rotation invariance
        
        Parameters:
        ----------
        img:[ndarray] 
        angle[int]: the radian value of rotation
        """
        img = img.rotate(angle)
        pic = np.array(img).astype('uint8')
        x = np.arange(224)
        y = np.arange(224)
        x,y = np.meshgrid(x,y)
        erea = (x-223/2)**2+(y-223/2)**2
        pic[erea>(self.rf_size/2)**2,:] = [127,127,127]
        return pic
    
    def show(self,pklfile,subfile):
        """
        pklefile[str]
        subfile[str]
        """
        path = os.getcwd()
        filepath = pjoin(path,'DataStore',pklfile)
        with open(filepath,'rb') as f:
            picdict = pickle.load(f)
            
        plt.imshow(picdict[subfile])
    
    def fopen(self,pklfile):
        path = pjoin(os.getcwd(),'DataStore')
        with open(pjoin(path,pklfile),'rb') as f:
            fdict = pickle.load(f)
        return fdict

    
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
            stimuli_dict = {}
            para_df = pd.DataFrame(columns=['lambda','factor','act'])
            for l_len in range(reg_lambda.size):
                for f_len in range(factor.size):
                    lam = reg_lambda[l_len]
                    fac = factor[f_len]
                    #certify img's stability
                    stimuli_set = np.random.rand(224,224,3).astype('uint8')[np.newaxis,:,:,:].transpose(0,3,1,2)
                    for run in range(nruns):
                        op_img = self.syn.synthesize(None, self.unit, 0.1, lam, 100,
                                                     '.', None, 0.2, fac, step=150)
                        op_img = ip.to_pil(op_img,True)
                        op_img = np.array(op_img).transpose(2,0,1)
                        img = op_img[np.newaxis,:,:,:]
                        stimuli_set = np.concatenate((stimuli_set,img),axis=0)
                    stimuli_set = np.delete(stimuli_set,0,axis=0)
                    stimuli_dict[f'{lam}_{fac}']=stimuli_set
                    act = self.dnn.compute_activation(stimuli_set, self.mask).get(self.layer)[:,0,self.unit[0],self.unit[1]]
                    act_sta = np.mean(np.array(act))
                    #add values to dataframe
                    info = pd.DataFrame({'lambda':lam,'factor':fac,'act':act_sta}, index=[0])
                    para_df = para_df.append(info, ignore_index=True)
            # sort the dataframe to get top parameters
            para_df.to_csv(pjoin(os.getcwd(),'Paras1.csv'))
            para_df = para_df.sort_values(by=['act'], ascending=False).reset_index(drop=True)
            para_df.to_csv(pjoin(os.getcwd(),'Paras2.csv'))
            para_act = para_df.iloc[0:top,:]
            # generate top parameters dict
            top_param = para_act.to_dict(orient='index')
            # store top_para as pickle
            cur_path = os.getcwd()
            
            stimuli_file = pjoin(cur_path,'stimuli.pickle') 
            with open(stimuli_file,'wb') as f:
                pickle.dump(stimuli_dict,f)
            
            store_folder = pjoin(cur_path, 'DataStore')
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            file = pjoin(store_folder, f'TopParameters-{self.layer}_{self.channel}.pickle')
            if os.path.exists(file):
                os.remove(file)            
            with open(file, 'wb') as f:
                pickle.dump(top_param, f)
            return top_param

    def gen_opt(self, top_param, subnum):
        """
        generate optimal images according to the given top parameters
        
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
                op_img = self.syn.synthesize(None, self.unit, 0.1, lam, 100,
                                             '.', None, 0.2, fac, step=150)
                # transpose its shape to (224,224,3)
                op_img = ip.to_pil(op_img,True)
                op_img = np.array(op_img)
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

    def gen_tran(self, optimal, axis, stype='optimal'):
        """
        Generate stimulus used in transition invariance expirement
        Only support the half of the original stimuli enter the Left margin of the receptive field
        to move to the symmetrical on the right in stride 1
        
        Parameters
        ----------
        optimal[dict]
        axis[str]: 'X' or 'Y'
        stype[str]: 'optimal' or 'natural'
            
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
        if not stype in ['optimal','natural'] :
            raise ValueError('stype only support optimal and natural!')
        # Run stimuli
        if not axis in ['X','Y'] :
            raise ValueError('axis only support X and Y!')
        else:
            opt_tran = dict()
            for org in optimal.keys():
                op_img = optimal[org]
                #crop op_img according to its rf_size
                center = int(224/2)
                span = round(self.rf_size/2)
                op_img = op_img[center-span:center+span+1, center-span:center+span+1,:]#.astype('uint8')
                #op_img = ip.to_pil(op_img.transpose(2,0,1))
                op_img = Image.fromarray(op_img)
                #move the oringal image to generate new stimulus
                center_new = int(448/2)
                start_y = round((448-self.rf_size)/2)
                start_x = 224-self.rf_size-1
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
                        tr_key = f'{org}_move:{move-224+span}'
                        opt_tran[tr_key] = op_trg
                #translate in axis Y
                else:
                    for move in range(start_x, center_new+1):
                        #init a background waiting for paste, its RGB:(127,127,127)
                        bkg = np.zeros((448,448,3),dtype=np.uint8)
                        bkg[:,:,:] = 127
                        bkg = Image.fromarray(bkg)
                        bkg.paste(op_img, (start_y, move))
                        op_trg = np.asarray(bkg.crop((112,112,336,336)))
                        #add vaues to dict
                        tr_key = f'{org}_move:{move-224+span}'
                        opt_tran[tr_key] = op_trg
            #save the dict using pickle
            cur_path = os.getcwd()
            if stype == 'natural':
                store_folder = pjoin(cur_path, 'NaDataStore')
            else:
                store_folder = pjoin(cur_path, 'DataStore')
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            file = pjoin(store_folder, f'TransferStimuli-{self.layer}_{self.channel}.pickle')
            if os.path.exists(file):
                os.remove(file)
            with open(file, 'wb') as f:
                pickle.dump(opt_tran, f)
            return opt_tran
    
    def gen_rot(self, optimal, interval, stype='optimal'):
        """
        Generate stimulus used in rotation invariance expirement
        
        Parameters
        ----------
        optimal[dict]
        interval[int] the interval of degrees
        stype[str]: 'optimal' or 'natural'
        
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
        if not stype in ['optimal','natural'] :
            raise ValueError('stype only support optimal and natural!')
        opt_rot = dict()
        for org in optimal.keys():
            op_img = optimal[org]#.astype('uint8')
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
        if stype == 'natural':
            store_folder = pjoin(cur_path, 'NaDataStore')
        else:
            store_folder = pjoin(cur_path, 'DataStore')
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        file = pjoin(store_folder, f'RotateStimuli-{self.layer}_{self.channel}.pickle')
        if os.path.exists(file):
            os.remove(file)            
        with open(file, 'wb') as f:
            pickle.dump(opt_rot, f)
        return opt_rot
    
    def gen_sca(self, optimal, num, stype='optimal'):
        """
        Generate stimulus used in scaling invariance expirement

        Parameters
        ----------
        optimal[dict]
        num[int] the total nums of scale stimuli
        stype[str]: 'optimal' or 'natural'
        
        Returns
        -------
        opt_sca[dict]
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
        if not stype in ['optimal','natural'] :
            raise ValueError('stype only support optimal and natural!')
        #init dict for recording info
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
            if stype == 'natural':
                store_folder = pjoin(cur_path, 'NaDataStore')
            else:
                store_folder = pjoin(cur_path, 'DataStore')
            if not os.path.exists(store_folder):
                os.makedirs(store_folder)
            file = pjoin(store_folder, f'ScaleStimuli-{self.layer}_{self.channel}.pickle')
            if os.path.exists(file):
                os.remove(file)
            with open(file, 'wb') as f:
                pickle.dump(opt_sca, f)
            return opt_sca

    def gen_na(self, stim, top):
        """
        Pick top natural images of the target unit in ImageNet2012 test data
        
        Parameters
        ----------
        stim[stim]:dnnbrain stim object
        dmask[csv]:info of interested units
        top[int]:the top num you focus on.
        
        Returns
        -------
        natural[dict]
        """
        natural_save = dict()
        natural = dict()
        intere = 3*top
        # Extract Activation
        activation = self.dnn.compute_activation(stim, self.mask)
        # Use Array_statistic in dnn.base to Do Max-pooling
        pooled_act = activation.pool('max').get(self.layer).flatten()
        # Do Sorting and Arg-sorting
        act_sort = np.argsort(-pooled_act, axis=0, kind='heapsort')
        act_sort = act_sort[0:intere]
        pooled_act = -np.sort(-pooled_act, axis=0, kind='heapsort')
        pooled_act = pooled_act[0:intere]
        # Set .stim.csv Activation Information
        channel_stim = copy.deepcopy(stim)
        channel_stim.set('stimID', stim.get('stimID')[act_sort])
        #get specific location of units
        act_top = self.dnn.compute_activation(channel_stim, self.mask).get(self.layer)
        loc = []
        for num in range(intere):
            act_map = act_top[num].squeeze(0)
            pos = np.unravel_index(np.argmax(act_map),act_map.shape)
            loc.append(pos)
        #crop natural image of its receptive field
        topnum = 0
        act_test = np.zeros((intere))
        for img_id in channel_stim.get('stimID'):
            image = np.asarray(Image.open(pjoin(stim.header['path'], img_id)).convert('RGB')).transpose(2,0,1)
            image = ip.resize(image, (224,224)).transpose(1,2,0)
            #init bkg
            bkg = np.zeros((224,224,3),dtype=np.uint8)
            bkg[:,:,:] = 127
            bkg = Image.fromarray(bkg)                
            #get units location
            na_unit = loc[topnum]
            #define para to pick region of rf
            center_x = int((na_unit[0]/self.fm_size) * 224)
            center_y = int((na_unit[1]/self.fm_size) * 224)
            threshold = int(self.rf_size/2)
            #define rule to handle different situation
            rule = lambda x,y: (x,y) if (x>0) & (y<224) else (0,self.rf_size) if x<0 else (224-self.rf_size,224)
            x_pos = rule(center_x-threshold, center_x+threshold)
            y_pos = rule(center_y-threshold, center_y+threshold)
            na_img = image[x_pos[0]:x_pos[1], y_pos[0]:y_pos[1], :]
            na_img = Image.fromarray(na_img)
            #get upper left pos and paste
            point = int(112-self.rf_size/2)
            bkg.paste(na_img, (point, point))
            nat_img = np.asarray(bkg)
            #compute act to select useful rf_image
            img_test = nat_img[np.newaxis,:,:,:].transpose(0,3,1,2)
            act_te = self.dnn.compute_activation(img_test, self.mask).get(self.layer)[0,0,6,6]
            act_test[topnum] = act_te
            #add values to dict
            nasave_key = f'{topnum}'
            natural_save[nasave_key] = nat_img
            topnum += 1
        #select images which act bias is less
        standard = np.mean(act_test)
        act_diff = act_test - standard
        interest = np.argsort(act_diff)[-top:][::-1]
        num = 0
        print(natural_save.keys())
        for item in interest:
            pic =  natural_save[str(item)]
            na_key = f'top{num+1}'
            natural[na_key] = pic
            num += 1
        #save the dict using pickle
        cur_path = os.getcwd()
        store_folder = pjoin(cur_path, 'NaDataStore')
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        file = pjoin(store_folder, f'NaturalStimuli-{self.layer}_{self.channel}.pickle')
        if os.path.exists(file):
            os.remove(file)
        with open(file, 'wb') as f:
            pickle.dump(natural, f)
        return natural

    def extr_act(self, stim, topnum, subnum, stype='optimal'):
        """
        process one stimuli dict to csv file

        Parameters
        ----------
        stim[dict/pickle]
        topnum [int]: num of parameters
        subnum [int]: num of repeats
        stype[str]: 'optimal' or 'natural'
        
        Returns
        -------
        act[csv]
            row_name:top_subnum  column_name:inv_index  value:dnn.act 

        """
        # Type check
        if not (isinstance(stim,str) or isinstance(stim,dict)):
            raise TypeError('Only dict or .pickle file path is available')
        if not stype in ['optimal','natural'] :
            raise ValueError('stype only support optimal and natural!')
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
        if stype == 'optimal':
            # extract activation of optimal
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
                    #generate activation dict
                    stimuli_set = np.delete(stimuli_set,0,axis=0)
                    activ = self.dnn.compute_activation(stimuli_set,self.mask).get(self.layer)[:,0,self.unit[0],self.unit[1]]
                    activ_list = list(activ)
                    activ_dict[picname]=activ_list
            #define store folder
            cur_path = os.getcwd()
            store_folder = pjoin(cur_path, 'ActData')
        else:
            na_top = topnum * subnum
            for num in range(na_top):
                picname = f'top{num+1}'
                activ_list = []
                column_list = []
                stimuli_set = np.random.rand(224,224,3).astype('uint8')[np.newaxis,:,:,:].transpose(0,3,1,2)
                for key in stim.keys():
                    time = eval(re.sub('\D','',key.split('_')[0]))
                    if num == time-1:
                        level = eval(key[key.rfind(':')+1:])
                        stimuli = stim[key]
                        dnn_input = stimuli[np.newaxis,:,:,:].transpose(0,3,1,2)
                        stimuli_set = np.concatenate((stimuli_set,dnn_input),axis=0)
                        column_list.append(level)
                #generate activation dict
                stimuli_set = np.delete(stimuli_set,0,axis=0)
                activ = self.dnn.compute_activation(stimuli_set,self.mask).get(self.layer)[:,0,self.unit[0],self.unit[1]]
                activ_list = list(activ)
                activ_dict[picname]=activ_list
            #define store folder
            cur_path = os.getcwd()
            store_folder = pjoin(cur_path, 'NaActData')
        #generate dataframe
        df_activ = pd.DataFrame(activ_dict)
        df_activ = pd.DataFrame(np.array(df_activ).transpose(),
                                index=list(activ_dict.keys()),columns=column_list)
        df_activ = df_activ.sort_index(axis=1)
        # save the df as csv
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        file_name = pjoin(store_folder,f'{self.layer}_{self.channel}_{type_name}.csv')
        df_activ.to_csv(file_name)
        