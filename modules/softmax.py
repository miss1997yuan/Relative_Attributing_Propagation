
import tensorflow as tf
from module import Module



class Softmax(Module):
    '''
    Softmax Layer
    '''

    def __init__(self, name='softmax'):
        self.name = name
        Module.__init__(self)
        

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        with tf.name_scope(self.name):
            #with tf.name_scope('activations'):
            self.activations = tf.nn.softmax(self.input_tensor, name=self.name)
            tf.summary.histogram('activations', self.activations)
        return self.activations

    def clean(self):
        self.activations = None

    def lrp(self,R,*args,**kwargs):
        # component-wise operations within this layer
        # ->
        # just propagate R further down.
        # makes sure subroutines never get called.
        self.R = R
        #Rx = self.input_tensor  * self.activations
        Rx = self.input_tensor  * self.R
        #Rx = Rx / tf.reduce_sum(self.input_tensor)
        
        #import pdb; pdb.set_trace()
        tf.summary.histogram(self.name, Rx)
        return Rx
    
    