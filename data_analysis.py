# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:36:57 2020

@author: lenovo
"""
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

class DataAnalysis:
    """
    Analysis stim's invariance according to the corresponding act csv
    """
    def __init__(self):
        pass
    
    
    
    def plot_inv(self, act):
        """
        Plot invariance curve

        Parameters
        ----------
        act[csv]

        Returns
        -------
        img[ndarray]

        """
        pass
        
    def plot_tun(self, act):
        """
        Plot tunning curve

        Parameters
        ----------
        act[csv]

        Returns
        -------
        img[ndarray]

        """
        pass