import tensorflow as tf


# -------------------------------
# Modules for the neural network
# -------------------------------
layer_count = 0
class Module:

    def __init__(self):
        global layer_count
        layer_count = layer_count + 1
        
        if hasattr(self, 'name'):
            self.name = self.name+'_'+str(layer_count)
        self.lrp_var = None
        self.lrp_param = 1.
        
    def forward(self,X):
        return X

    def clean(self):
        pass

    def set_lrp_parameters(self,lrp_var=None,param=None):
        self.lrp_var = lrp_var
        self.lrp_param = param

    def lrp(self,R, lrp_var=None,param=None):

        if lrp_var == None and param == None:
            # module.lrp(R) has been called without further parameters.
            # set default values / preset values
            lrp_var = self.lrp_var
            param = self.lrp_param

        if lrp_var.lower() == 'lrp' or lrp_var.lower() == 'LRP':
            return self._simple_lrp(R)
        elif lrp_var.lower() == 'dtd' or lrp_var.lower() == 'DTD':
            return self._simple_deep_lrp(R)
        elif lrp_var.lower() == 'lrpab' or lrp_var.lower() == 'LRPab':
            return self._alphabeta_deep_lrp(R, param)
        # elif lrp_var.lower() == 'rap' or lrp_var.lower() == 'RAP':
        #     return self._RAP(R)
        else:
            print ('Unknown lrp variant', lrp_var)
    def RAP(self,R, lrp_var=None,param=None):


        return self._RAP(R, R)
