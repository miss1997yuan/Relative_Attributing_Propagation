

import tensorflow as tf
from module import Module


from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import nn_ops, gen_nn_ops
import numpy as np
class MaxPool(Module):

    def __init__(self, pool_size=2, pool_stride=None, pad = 'SAME',name='maxpool'):
        self.name = name
        Module.__init__(self)
        self.pool_size = pool_size
        self.pool_kernel = [1, self.pool_size, self.pool_size, 1]
        self.pool_stride = pool_stride
        if self.pool_stride is None:
            self.stride_size=self.pool_size
            self.pool_stride=[1, self.stride_size, self.stride_size, 1] 
        self.pad = pad
        

    def forward(self, input_tensor, batch_size=10):
        # self.input_tensor = input_tensor
        self.input_tensor = input_tensor
        self.in_N, self.in_h, self.in_w, self.in_depth = self.input_tensor.get_shape().as_list()

        # with tf.variable_scope(self.name):
        with tf.name_scope(self.name):
            self.activations = tf.nn.max_pool(self.input_tensor, ksize=self.pool_kernel, strides=self.pool_stride,
                                              padding=self.pad, name=self.name)
            tf.summary.histogram('activations', self.activations)
        X = self.activations

        return X

    def clean(self):
        self.activations = None
        self.R = None


    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        self.check_shape(R)
        if self.R.shape[1] == 1:
            self.R = tf.reshape(self.R,[self.batch_size,7,7,512])
        Z = tf.nn.max_pool(self.input_tensor, ksize=self.pool_kernel, strides=self.pool_stride, padding='SAME') + 1e-9
        S = self.R / Z
        C = gen_nn_ops.max_pool_grad(self.input_tensor, Z, S, ksize=self.pool_kernel, strides=self.pool_stride,
                                        padding='SAME')
        result = self.input_tensor * C
        return result

    def _RAP(self,R_p, R_n):
        self.check_shape_RAP(R_p, True)
        self.check_shape_RAP(R_n, False)

        _, mask = tf.nn.max_pool_with_argmax(self.input_tensor, ksize=self.pool_kernel, strides=self.pool_stride, padding='SAME')
        _, mask2 = tf.nn.max_pool_with_argmax(self.input_tensor, ksize=self.pool_kernel, strides=self.pool_stride, padding='SAME')
        result_plus = self.unpool(tf.maximum(0.0,self.R_p), mask, 2)
        result_minus = self.unpool(tf.minimum(0.0,self.R_p), mask2, 2)

        res_pos = result_plus + result_minus

        # _, mask = tf.nn.max_pool_with_argmax(self.input_tensor, ksize=self.pool_kernel, strides=self.pool_stride,
        #                                      padding='SAME')
        # _, mask2 = tf.nn.max_pool_with_argmax(self.input_tensor, ksize=self.pool_kernel, strides=self.pool_stride,
        #                                       padding='SAME')
        result_plus = self.unpool(tf.maximum(0.0, self.R_n), mask, 2)
        result_minus = self.unpool(tf.minimum(0.0, self.R_n), mask2, 2)

        res_neg = result_plus + result_minus

        return res_pos, res_neg

    def unpool(self,att, mask, stride):
        assert mask is not None
        ksize = [1, stride, stride, 1]
        input_shape = att.get_shape().as_list()
        #  calculation new shape
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int64)
        f = one_like_mask * feature_range
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(att)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(att, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret
    def _simple_deep_lrp(self,R):
        return self._simple_lrp(R)
    def _alphabeta_deep_lrp(self,R,alpha):
        return self._simple_lrp(R)

    def check_shape(self, R):
        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape) != 4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, activations_shape)
        N,self.Hout,self.Wout,NF = self.R.get_shape().as_list()

    def check_shape_RAP(self, R, tag):
        if tag == True:
            self.R_p = R
            R_shape = self.R_p.get_shape().as_list()
            if len(R_shape) != 4:
                activations_shape = self.activations.get_shape().as_list()
                self.R_p = tf.reshape(self.R_p, activations_shape)
            N, self.Hout, self.Wout, NF = self.R_p.get_shape().as_list()
        else:
            self.R_n = R
            R_shape = self.R_n.get_shape().as_list()
            if len(R_shape) != 4:
                activations_shape = self.activations.get_shape().as_list()
                self.R_n = tf.reshape(self.R_n, activations_shape)
            N, self.Hout, self.Wout, NF = self.R_n.get_shape().as_list()


