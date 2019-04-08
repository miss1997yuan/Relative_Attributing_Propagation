

import tensorflow as tf


class Train():
    def __init__(self, output=None,ground_truth=None,loss='sparse_crossentropy', optimizer='Adam', opt_params=[]):
        self.output = output
        self.ground_truth = ground_truth
        self.loss = loss
        self.optimizer = optimizer
        self.opt_params = opt_params

        self.learning_rate = self.opt_params[0]
        if len(self.opt_params)>1:
            self.var_list = self.opt_params[1]
        else:
            self.var_list = None

        if type(self.loss)!=str:
            #assuming loss is already computed and passed as a tensor
            self.cost = tf.reduce_mean(self.loss)
            #tf.summary.scalar('Loss', self.cost)
        else:
            self.compute_cost()
        
        self.optimize()
        
    def compute_cost(self):

        # Available loss operations are - [softmax_crossentropy, sigmoid_crossentropy, MSE] 
        if self.loss=='softmax_crossentropy':
            #Cross Entropy loss:
            with tf.name_scope('cross_entropy'):
                diff = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.ground_truth)
                self.cost = tf.reduce_mean(diff)
            tf.summary.scalar('Loss', self.cost)
        elif self.loss=='sparse_crossentropy':

            with tf.name_scope('cross_entropy'):

                diff = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.ground_truth)
                self.cost = tf.reduce_mean(diff)
            tf.summary.scalar('Loss', self.cost)
        elif self.loss=='sigmoid_crossentropy':
            #Cross Entropy loss:
            with tf.name_scope('cross_entropy'):
                diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.ground_truth)
                self.cost = tf.reduce_mean(diff)
            tf.summary.scalar('Loss', self.cost)
        elif self.loss=='focal loss':
            with tf.name_scope('cross_entropy'):
                alpha = 0.25
                gamma = 2.0
                logits = tf.convert_to_tensor(tf.nn.softmax(self.output))
                onehot_labels = tf.convert_to_tensor(self.ground_truth)

                precise_logits = tf.cast(logits, tf.float32) if (
                    logits.dtype == tf.float16) else logits
                onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
                predictions = tf.nn.sigmoid(precise_logits)
                predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1. - predictions)
                # add small value to avoid 0
                epsilon = 1e-8
                alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
                alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1 - alpha_t)
                self.cost = tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt + epsilon),
                                       axis=1)
            #tf.summary.scalar('Loss', self.cost)
        elif self.loss=='MSE':
            with tf.name_scope('mse_loss'):
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.output, self.ground_truth))))
            tf.summary.scalar('Loss', self.cost)
        else:
            print ('Loss should be one of [softmax_crossentropy, sparse_crossentropy, focal loss, sigmoid_crossentropy, MSE] ')
            print ('If not define your own loss')
        
    

    def optimize(self):
        # Available loss operations are - [adam, adagrad, adadelta, grad_descent, rmsprop]
        with tf.name_scope('train'):
            if self.optimizer == 'adam':
                self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            elif self.optimizer == 'meprop':
                tmp_train = tf.train.AdamOptimizer(self.learning_rate)
                cg = tmp_train.compute_gradients(self.cost)
                mask = []

                for cnt in range(len(cg)-1):

                    if cg[cnt][0] != None:
                        arr, v = cg[cnt]
                        if arr.shape.ndims == 1:
                            mask.append(cg[cnt])
                        else:
                            K = int(arr.shape[-1]/8)
                            values, indices = tf.nn.top_k(abs(arr), k=K, sorted=True)

                            temp_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(
                                tf.shape(arr)[:(arr.get_shape().ndims - 1)]) + [K])], indexing='ij')
                            temp_indices = tf.stack(temp_indices[:-1] + [indices], axis=-1)
                            values = tf.reshape(tf.gather_nd(arr, temp_indices), [-1])
                            full_indices = tf.reshape(temp_indices, [-1, arr.get_shape().ndims])
                            # values = tf.reshape(values, [-1])

                            mask_st = tf.SparseTensor(indices=tf.cast(
                                full_indices, dtype=tf.int64), values=values, dense_shape=arr.shape)
                            mask.append((tf.sparse_tensor_to_dense(tf.sparse_reorder(mask_st)),v))


                    else:
                        mask.append(cg[cnt])
                mask.append(cg[-1])

                self.train = tmp_train.apply_gradients(mask)
            elif self.optimizer == 'rmsprop':
                self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            elif self.optimizer == 'grad_descent':
                self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
            elif self.optimizer == 'adagrad':
                self.train = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.optimizer == 'adadelta':
                self.train = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.cost)
            else:
                print ('Optimizer should be one of: [adam, meprop, adagrad, adadelta, grad_descent, rmsprop]')

                
