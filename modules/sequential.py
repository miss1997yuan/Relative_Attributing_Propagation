

import copy
import sys
import numpy as np
from module import Module
from train import Train
na = np.newaxis
import tensorflow as tf
# -------------------------------
# Sequential layer
# -------------------------------
class Sequential(Module):


    def __init__(self,modules):

        Module.__init__(self)
        self.modules = modules


    def forward(self,X):

        if 'conv' in self.modules[0].name:
            if self.modules[0].batch_size is None or self.modules[0].input_depth is None or self.modules[0].input_dim is None:
                raise ValueError('Expects batch_input_shape= AND input_depth= AND input_dim= for the first layer ')
        elif 'linear' in self.modules[0].name:
            if self.modules[0].batch_size is None or self.modules[0].input_dim is None:
                raise ValueError('Expects batch_input_shape= AND input_dim= for the first layer ')
        
        
        print ('Forward Pass ... ')
        print ('------------------------------------------------- ')
        for m in self.modules:
            m.batch_size=self.modules[0].batch_size
            print (m.name+'::',)
            print (X.get_shape().as_list())
            
            X = m.forward(X)
            
        print ('\n'+ '------------------------------------------------- ')
        
        return X


    def clean(self):

        for m in self.modules:
            m.clean()



    def lrp(self,R,lrp_var=None,param=None):

        print ('Computing Relevance ... ')
        print ('------------------------------------------------- ')

        for m in self.modules[::-1]:
            R = m.lrp(R,lrp_var,param)
            print (m.name+'::',)
            print (R.get_shape().as_list())
            
        print ('\n'+'------------------------------------------------- ')

        return R

    def lrp_layerwise(self, m, R,lrp_var=None,param=None):
        R = m.lrp(R,lrp_var,param)

        print (m.name+'::',)
        m.clean()
        print (R.get_shape().as_list())
        return R

    def RAP(self, R_pos, R_neg):

        print('Computing Relevance ... ')
        print('------------------------------------------------- ')

        for m in self.modules[::-1]:
            R_pos, R_neg = m.RAP(R_pos, R_neg)
            print(m.name + '::', )
            print(R_pos.get_shape().as_list())

        print('\n' + '------------------------------------------------- ')

        return R_pos, R_neg

    def RAP_layerwise(self, m, R_pos, R_neg):
        R_pos, R_neg = m.RAP(R_pos, R_neg)

        print(m.name + '::', )
        m.clean()
        print(R_pos.get_shape().as_list())
        return R_pos, R_neg
    def fit(self,output=None,ground_truth=None,loss='CE', optimizer='Adam', opt_params=[]):
        return Train(output,ground_truth, loss, optimizer, opt_params)
