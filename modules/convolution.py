
import pdb
from math import ceil

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import nn_ops, gen_nn_ops
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import activations
import variables
from module import Module
import numpy as np

class Convolution(Module):
    '''
    Convolutional Layer
    '''

    def __init__(self, output_depth, batch_size=None, input_dim = None, input_depth=None, kernel_size=5, stride_size=2, act = 'relu', phrase=True, pad = 'SAME', weights_init= tf.truncated_normal_initializer(stddev=0.01), bias_init= tf.constant_initializer(0.0), name="conv2d",first = False, final = False):
        self.name = name
        #self.input_tensor = input_tensor
        Module.__init__(self)
        
        
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_depth = input_depth
        
        self.final_layer = final
        self.first_layer = first
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.act = act
        self.phrase = phrase
        self.pad = pad

        self.training = phrase
        self.momentum = 0.9
        self.epsilon = 0.001
        self.weights_init = weights_init
        self.bias_init = bias_init

        

    def check_input_shape(self):
        inp_shape = self.input_tensor.get_shape().as_list()
        try:
            if len(inp_shape)!=4:
                mod_shape = [self.batch_size, self.input_dim,self.input_dim,self.input_depth]
                self.input_tensor = tf.reshape(self.input_tensor, mod_shape)
        except:
            raise ValueError('Expected dimension of input tensor: 4')


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.strides = [1, self.stride_size, self.stride_size, 1]
        with tf.variable_scope(self.name):
            data_dict = np.load("./vgg16.npy", encoding='latin1').item()
            list = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8']
            # data_dict[list[int(self.name[-1])]][0]
            num = int(self.name[7:])
            if num < 4:
                num = num - 1
            elif num>=4 and num <7:
                num = num - 2
            elif num >= 7 and num < 11:
                num = num -3
            elif num >= 11 and num <15:
                num = num - 4
            elif num >= 15 and num <19:
                num = num - 5
            else:
                num = num - 6
            self.l_num = num
            self.weights = tf.constant(data_dict[list[num]][0], name="filter")

            self.biases = tf.constant(data_dict[list[num]][1], name="biases")
            if num!=0:
                self.prev_biases = tf.constant(data_dict[list[num-1]][1], name="biases")
        with tf.name_scope(self.name):
            with tf.name_scope(self.name):

                if self.final_layer == False:
                    if list[num] == 'fc6' or list[num] == 'fc7':
                        self.input_tensor = tf.reshape(self.input_tensor, [self.batch_size, -1])
                        conv = tf.matmul(self.input_tensor, self.weights)
                        self.conv_biases = tf.nn.bias_add(conv, self.biases)

                    else:
                        conv = tf.nn.conv2d(self.input_tensor, self.weights, strides=self.strides, padding=self.pad)
                        self.conv = conv
                        self.conv_biases = tf.nn.bias_add(conv, self.biases)
                    if isinstance(self.act, str):
                        self.activations = tf.nn.relu(self.conv_biases)
                    elif hasattr(self.act, '__call__'):
                        self.activations = self.act(self.conv_biases)
                    print('not final')

                else:
                    if list[num] == 'fc8':
                        self.input_tensor = tf.reshape(self.input_tensor, [self.batch_size, -1])
                        conv = tf.matmul(self.input_tensor, self.weights)
                        self.conv = conv
                        self.conv_biases = tf.nn.bias_add(conv, self.biases)

                        self.activations = self.conv_biases

                    print('final')
                tf.summary.histogram('activations', self.activations)
                tf.summary.histogram('weights', self.weights)
                tf.summary.histogram('biases', self.biases)

        return self.activations

    def _simple_lrp(self,R):
        self.R = R
        if len(self.R.shape) == 2:
            self.R = tf.expand_dims(tf.expand_dims(self.R, 1), 1)
        if len(self.weights.shape) == 2:
            self.weights = tf.expand_dims(tf.expand_dims(self.weights, 0), 0)
        if len(self.input_tensor.shape) == 2:
            self.input_tensor = tf.expand_dims(tf.expand_dims(self.input_tensor, 1), 1)
        if  self.weights.shape[2] == 25088:
            self.weights = tf.reshape(self.weights, [7, 7, 512, 4096])
            self.input_tensor = tf.reshape(self.input_tensor, [10, 7, 7, 512])
        # self.R = R

        tmp_weight = tf.maximum(0.0,self.weights)
        X = self.input_tensor
        Z = tf.nn.conv2d(X, tmp_weight, strides=self.strides, padding=self.pad) + 1e-9
        S = self.R/Z
        result = X * (
        nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), tmp_weight, S, strides=self.strides,
                                     padding=self.pad))
        return result
    def _simple_deep_lrp(self,R):

        self.R = R
        if len(self.R.shape) == 2:
            self.R = tf.expand_dims(tf.expand_dims(self.R, 1), 1)
        if len(self.weights.shape) == 2:
            self.weights = tf.expand_dims(tf.expand_dims(self.weights, 0), 0)
        if len(self.input_tensor.shape) == 2:
            self.input_tensor = tf.expand_dims(tf.expand_dims(self.input_tensor, 1), 1)
        if  self.weights.shape[2] == 25088:
            self.weights = tf.reshape(self.weights, [7, 7, 512, 4096])
            self.input_tensor = tf.reshape(self.input_tensor, [10, 7, 7, 512])
        if self.first_layer == True:
            pweight = tf.maximum(1e-9, self.weights)
            nweight = tf.minimum(-1e-9, self.weights)
            X = self.input_tensor
            L = self.input_tensor*0+ tf.reduce_max(self.input_tensor, [1, 2, 3], keep_dims=True)
            H = self.input_tensor*0+ tf.reduce_min(self.input_tensor,[1,2,3],keep_dims=True)
            Z = tf.nn.conv2d(X, self.weights, strides=self.strides, padding=self.pad)\
                - tf.nn.conv2d(L, pweight, strides=self.strides, padding=self.pad)\
                -tf.nn.conv2d(H, nweight, strides=self.strides, padding=self.pad)+1e-9
            S = self.R/Z
            result = X*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), self.weights, S, strides=self.strides, padding=self.pad))\
                     -L*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, S, strides=self.strides,padding=self.pad))-\
                     H*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, S, strides=self.strides, padding=self.pad))

        else:
            tmp_weight = tf.maximum(0.0,self.weights)
            X = self.input_tensor
            Z = tf.nn.conv2d(X, tmp_weight, strides=self.strides, padding=self.pad) + 1e-9
            S = self.R/Z
            result = X * (
            nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), tmp_weight, S, strides=self.strides,
                                         padding=self.pad))
        return result


    def _alphabeta_deep_lrp(self, R, alpha):


        alpha = 2
        beta = 1
        self.R = R
        if len(self.R.shape) == 2:
            self.R = tf.expand_dims(tf.expand_dims(self.R, 1), 1)
        if len(self.weights.shape) == 2:
            self.weights = tf.expand_dims(tf.expand_dims(self.weights, 0), 0)
        if len(self.input_tensor.shape) == 2:
            self.input_tensor = tf.expand_dims(tf.expand_dims(self.input_tensor, 1), 1)
        if  self.weights.shape[2] == 25088:
            self.weights = tf.reshape(self.weights, [7, 7, 512, 4096])
            self.input_tensor = tf.reshape(self.input_tensor, [10, 7, 7, 512])

        pweight = tf.maximum(1e-9, self.weights)
        nweight = tf.minimum(-1e-9, self.weights)

        if self.first_layer == True:
            X = self.input_tensor
            L = self.input_tensor*0 + tf.reduce_min(self.input_tensor, [1, 2, 3], keep_dims=True)
            H = self.input_tensor*0 + tf.reduce_max(self.input_tensor, [1, 2, 3], keep_dims=True)
            Z = tf.nn.conv2d(X, self.weights, strides=self.strides, padding=self.pad)\
                - tf.nn.conv2d(L, pweight, strides=self.strides, padding=self.pad)\
                -tf.nn.conv2d(H, nweight, strides=self.strides, padding=self.pad)+1e-9
            S = self.R/Z
            result = X*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), self.weights, S, strides=self.strides, padding=self.pad))\
                     -L*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, S, strides=self.strides,padding=self.pad))-\
                     H*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, S, strides=self.strides, padding=self.pad))

        else:
            X = self.input_tensor + 1e-9
            Za = tf.nn.conv2d(X, pweight, strides=self.strides, padding=self.pad)
            Sa = alpha*self.R/Za
            Zb = tf.nn.conv2d(X, nweight, strides=self.strides, padding=self.pad)
            Sb = -beta * self.R / Zb
            result = X*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, Sa, strides=self.strides, padding=self.pad)
            +nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, Sb, strides=self.strides, padding=self.pad))
        return result
    # def __simple_lrp_attention(self,R, att_plus, att_minu):
    #     self.R = R
    #     # self.check_shape(R)
    #     if len(self.R.shape) == 2:
    #         self.R = tf.expand_dims(tf.expand_dims(self.R, 1), 1)
    #     if len(self.weights.shape) == 2:
    #         self.weights = tf.expand_dims(tf.expand_dims(self.weights, 0), 0)
    #     if len(self.input_tensor.shape) == 2:
    #         self.input_tensor = tf.expand_dims(tf.expand_dims(self.input_tensor, 1), 1)
    #     if  self.weights.shape[2] == 25088:
    #         self.weights = tf.reshape(self.weights, [7, 7, 512, 4096])
    #         self.input_tensor = tf.reshape(self.input_tensor, [10, 7, 7, 512])
    #
    #     if self.first_layer == True:
    #         pweight = tf.maximum(1e-9, self.weights)
    #         nweight = tf.minimum(-1e-9, self.weights)
    #         X = self.input_tensor
    #         L = self.input_tensor * 0 + tf.reduce_max(self.input_tensor, [1, 2, 3], keep_dims=True)
    #         H = self.input_tensor * 0 + tf.reduce_min(self.input_tensor,[1,2,3],keep_dims=True)
    #         Z = tf.nn.conv2d(X, self.weights, strides=self.strides, padding=self.pad)\
    #             - tf.nn.conv2d(L, pweight, strides=self.strides, padding=self.pad)\
    #             - tf.nn.conv2d(H, nweight, strides=self.strides, padding=self.pad)+1e-9
    #         S1 = tf.maximum(0.0,self.R)/Z
    #         result1 = X*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), self.weights, S1, strides=self.strides, padding=self.pad))\
    #                  -L*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, S1, strides=self.strides,padding=self.pad))-\
    #                  H*(nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, S1, strides=self.strides, padding=self.pad))
    #         S2 = tf.minimum(0.0,self.R) / Z
    #         result2 = X * (
    #         nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), self.weights, S2, strides=self.strides,
    #                                      padding=self.pad)) \
    #                  - L * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, S2, strides=self.strides,
    #                                                      padding=self.pad)) - \
    #                  H * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, S2, strides=self.strides,
    #                                                    padding=self.pad))
    #
    #         res = result1 + result2
    #         a_plus = tf.maximum(1e-9, result1)
    #         a_minu = tf.minimum(-1e-9, result2)
    #     elif self.final_layer == True:
    #         pweight = tf.maximum(1e-9, self.weights)
    #         nweight = tf.minimum(-1e-9, self.weights)
    #
    #
    #         X = self.input_tensor + 1e-9
    #         Za = tf.nn.conv2d(X, pweight, strides=self.strides, padding=self.pad)
    #         Zb = tf.nn.conv2d(X, nweight, strides=self.strides, padding=self.pad)
    #
    #         Z =  tf.nn.conv2d(X, self.weights, strides=self.strides, padding=self.pad)
    #
    #         Sa = self.R / Za
    #
    #         Sb = self.R / Zb
    #         S = self.R/Z
    #         resultA = X * (tf.nn.conv2d_transpose(S, pweight, tf.shape(self.input_tensor), strides=self.strides,
    #                                                     padding=self.pad))
    #         resultB = X * (tf.nn.conv2d_transpose(S, nweight, tf.shape(self.input_tensor), strides=self.strides,
    #                                                     padding=self.pad))
    #
    #         result_pos = tf.maximum(1e-9, resultA) + tf.maximum(1e-9, resultB)
    #         result_neg = tf.minimum(1e-9, resultA) + tf.minimum(1e-9, resultB)
    #
    #         res = result_pos + result_neg
    #         a_plus = tf.reduce_sum(result_pos, -1, keep_dims=True)
    #         a_minu = tf.reduce_sum(result_neg, -1, keep_dims=True)
    #     else:
    #         pweight = tf.maximum(0.0, self.weights)
    #         nweight = tf.minimum(0.0, self.weights)
    #         X = self.input_tensor
    #
    #         Za = tf.nn.conv2d(X, pweight, strides=self.strides, padding=self.pad)
    #         Zb = tf.nn.conv2d(X, nweight, strides=self.strides, padding=self.pad)
    #         Z = tf.nn.conv2d(X, self.weights, strides=self.strides, padding=self.pad)
    #         # Za_sum = tf.reduce_sum(tf.maximum(0.0, Za), [1, 2, 3], keep_dims=True) + tf.reduce_sum(tf.maximum(0.0, Zb),
    #         #                                                                                        [1, 2, 3],
    #         #                                                                                        keep_dims=True)
    #         # Zb_sum = tf.reduce_sum(tf.minimum(0.0, Zb), [1, 2, 3], keep_dims=True) + tf.reduce_sum(tf.minimum(0.0, Za),
    #         #                                                                                        [1, 2, 3],
    #         #                                                                                        keep_dims=True)
    #         # Z_sum = tf.subtract(Za_sum, Zb_sum)
    #
    #         # Plus
    #         A_p = tf.maximum(0.0,self.R)
    #         A_n = tf.minimum(0.0, self.R)
    #         S1_a = A_p / Za #>0
    #         S1_b = A_p / Zb #<0
    #         S1 = A_p/Z
    #         result1_A = X * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, S1_a, strides=self.strides,
    #                                          padding=self.pad)) #>0
    #         result1_B = X * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, S1_b, strides=self.strides,
    #                                                      padding=self.pad)) #<0
    #         # minus
    #
    #         S2_a = A_n / Za #<0
    #         S2_b = A_n / Zb #>0
    #         S2 = A_n/Z
    #         result2_A = X * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, S2_a, strides=self.strides,
    #                                          padding=self.pad)) #<0
    #         result2_B = X * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, S2_b, strides=self.strides,
    #                                      padding=self.pad)) #>0
    #         result_pos = result1_A - result2_A
    #         result_neg = result1_B - result2_B
    #         # result1_sum = tf.reduce_sum(result1_A, [1, 2, 3], keep_dims=True) + tf.reduce_sum(result1_B, [1, 2, 3], keep_dims=True)
    #         # result2_sum = tf.reduce_sum(result2_A, [1, 2, 3], keep_dims=True) + tf.reduce_sum(result2_B, [1, 2, 3],
    #         #                                                                                   keep_dims=True)
    #         # result_pos = ((tf.maximum(0.0,result1_A)+tf.maximum(0.0,result1_B))*result1_sum + (tf.maximum(0.0,result2_A)+tf.maximum(0.0,result2_B)) * result2_sum)  / tf.add(result1_sum,result2_sum)
    #         # result_neg = ((tf.minimum(0.0,result1_A)+tf.minimum(0.0,result1_B))*result1_sum + (tf.minimum(0.0,result2_A)+tf.minimum(0.0,result2_B)) * result2_sum)  / tf.add(result1_sum,result2_sum)
    #         res = result_pos - result_neg
    #
    #         a_plus = tf.reduce_sum(result_pos, -1, keep_dims=True)
    #         a_minu = tf.reduce_sum(result_neg, -1, keep_dims=True)
    #         # a_plus = tf.minimum(0.0,result1_A)
    #         # a_minu = tf.maximum(0.0,result2_A)
    #
    #     return res
    def _RAP(self,R_p,R_n):


        self.R_p = R_p
        self.R_n = R_n
        if len(self.R_p.shape) == 2:
            self.R_p = tf.expand_dims(tf.expand_dims(self.R_p, 1), 1)
            self.R_n = tf.expand_dims(tf.expand_dims(self.R_n, 1), 1)
        if len(self.weights.shape) == 2:
            self.weights = tf.expand_dims(tf.expand_dims(self.weights, 0), 0)
        if len(self.input_tensor.shape) == 2:
            self.input_tensor = tf.expand_dims(tf.expand_dims(self.input_tensor, 1), 1)
        if  self.weights.shape[2] == 25088:
            self.weights = tf.reshape(self.weights, [7, 7, 512, 4096])
            self.input_tensor = tf.reshape(self.input_tensor, [10, 7, 7, 512])

        if self.first_layer == True:
            pweight = tf.maximum(1e-9, self.weights)
            nweight = tf.minimum(-1e-9, self.weights)
            X = self.input_tensor
            L = self.input_tensor * 0 + tf.reduce_max(self.input_tensor, [1, 2, 3], keep_dims=True)
            H = self.input_tensor * 0 + tf.reduce_min(self.input_tensor, [1, 2, 3], keep_dims=True)
            Z = tf.nn.conv2d(X, self.weights, strides=self.strides, padding=self.pad) \
                - tf.nn.conv2d(L, pweight, strides=self.strides, padding=self.pad) \
                - tf.nn.conv2d(H, nweight, strides=self.strides, padding=self.pad) + 1e-9
            S_p = self.R_p / Z
            S_n = self.R_n / Z
            res_pos = X * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), self.weights, S_p, strides=self.strides,
                                                    padding=self.pad)) \
                  - L * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, S_p, strides=self.strides,
                                                      padding=self.pad)) - \
                  H * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, S_p, strides=self.strides,
                                                    padding=self.pad))
            res_neg = X * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), self.weights, S_n, strides=self.strides,
                                             padding=self.pad)) \
                      - L * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, S_n, strides=self.strides,
                                                          padding=self.pad)) - \
                      H * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, S_n, strides=self.strides,
                                                        padding=self.pad))

        elif self.final_layer == True:
            self.R = self.R_p
            pweight = tf.maximum(0.0, self.weights)
            nweight = tf.minimum(0.0, self.weights)
            X = self.input_tensor+1e-9
            Za = tf.nn.conv2d(X, pweight, strides=self.strides, padding=self.pad)
            Zb = tf.nn.conv2d(X, nweight, strides=self.strides, padding=self.pad)
            Z =  tf.nn.conv2d(X, self.weights, strides=self.strides, padding=self.pad)
            Sa = self.R / Za
            Sb = self.R / Zb
            S = self.R/Z
            resultA = X * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), pweight, Sa, strides=self.strides,
                                                       padding=self.pad))
            resultB = -X * (nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), nweight, Sb, strides=self.strides,
                                                         padding=self.pad))
            # resultA = X * (tf.nn.conv2d_transpose(Sa, pweight, tf.shape(self.input_tensor), strides=self.strides,
            #                                             padding=self.pad))
            # resultB = -X * (tf.nn.conv2d_transpose(Sb, nweight, tf.shape(self.input_tensor), strides=self.strides,
            #                                             padding=self.pad))

            result_pos = tf.maximum(0.0, resultA) + tf.maximum(0.0, resultB)
            result_neg = tf.minimum(0.0, resultA) + tf.minimum(0.0, resultB)
            res = result_pos + result_neg
            res_pos = res
            res_neg = res
        else:

            f_w = self.R_p.shape[1].value*self.R_p.shape[1].value

            self.biases = self.biases - tf.ones_like(self.biases)*tf.reduce_mean(self.biases)
            div = tf.cast(tf.square(f_w),dtype="float32")
            self.biases = self.biases/div
            self.R_p = R_p - self.biases
            self.R_n = R_n + self.biases

            tmp_weight = tf.maximum(0.0, self.weights)
            X = self.input_tensor
            Z = tf.nn.conv2d(X, tmp_weight, strides=self.strides, padding=self.pad)
            S = self.R_p / Z
            result = X * (
                nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), tmp_weight, S, strides=self.strides,
                                             padding=self.pad))

            tmp_weight2 = tf.minimum(0.0, self.weights)
            X = self.input_tensor
            Z = tf.nn.conv2d(X, tmp_weight2, strides=self.strides, padding=self.pad)
            S = self.R_n / Z
            result2 = X * (
                nn_ops.conv2d_backprop_input(tf.shape(self.input_tensor), tmp_weight2, S, strides=self.strides,
                                             padding=self.pad))
            res_pos = result
            res_neg = result2


        return res_pos, res_neg


