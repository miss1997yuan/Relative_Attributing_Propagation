'''
@author: WooJeoung Nam
@author: Jaesik Choi
@author: SeongWhan Lee
@maintainer: WooJeoung Nam
@contact: nwj0612@korea.ac.kr
@date: 8.4.2019
@version: 1.0
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('.')
from modules.sequential import Sequential
from modules.convolution import Convolution
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import modules.render as render
import matplotlib.pyplot as plt
import imageio
import tensorflow as tf

from glob import glob
import numpy as np
import utils_vgg

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 10, 'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 5, 'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01, 'Initial learning rate')
flags.DEFINE_string("data_dir", 'data', 'Directory for storing data')
flags.DEFINE_string("summaries_dir", 'image_convolutional_logs', 'Summaries directory')
flags.DEFINE_boolean("relevance", True, 'Compute relevances')
flags.DEFINE_string("relevance_method", 'RAP', 'relevance methods: LRP/LRPab/DTD/RAP')
flags.DEFINE_boolean("save_model", False, 'Save the trained model')
flags.DEFINE_boolean("reload_model", False, 'Restore the trained model')
flags.DEFINE_integer("Class", 1000, 'Number of class.')
flags.DEFINE_integer("m_tag", 1, 'Number of class.')
flags.DEFINE_integer("path", 0, 'path')
FLAGS = flags.FLAGS


def visualize(relevances, images_tensor):
    n = FLAGS.batch_size
    heatmap = np.sum(relevances.reshape([n, 224, 224, 3]),axis=3)
    input_images = images_tensor.reshape([n, 224, 224, 3])
    heatmaps = []
    for h, heat in enumerate(heatmap):
        input_image = input_images[h]
        maps = render.hm_to_rgb(heat, input_image, scaling=3, sigma=1,cmap = 'seismic')
        heatmaps.append(maps)
        imageio.imsave('./result/imgnet/'+FLAGS.relevance_method+'/heatmap' + str(h) + '.jpg', maps,vmax=1,vmin=-1)
        imageio.imsave(
            './result/imgnet/' + FLAGS.relevance_method + '/img' + str(h) + '.jpg', render.enlarge_image(input_image,3))
    np.save('./result/imgnet/' + FLAGS.relevance_method + '/' + '_heatmap.npy', relevances)
    R = np.array(heatmaps)
    with tf.name_scope('input_reshape'):
        img = tf.summary.image('input', tf.cast(R, tf.float32), n)
    return img.eval()


def plot_relevances(rel, img, writer):
    img_summary = visualize(rel, img)
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0, len(data))
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def nn():
    return Sequential(
        [Convolution(kernel_size=3, output_depth=64, input_depth=3, batch_size=FLAGS.batch_size, input_dim=3,
                     act='relu', stride_size=1, pad='SAME', first = True),
         Convolution(kernel_size=3, output_depth=64, input_depth=64, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         MaxPool(),

         Convolution(kernel_size=3, output_depth=128, input_depth=64, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         Convolution(kernel_size=3, output_depth=128, input_depth=128, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         MaxPool(),

         Convolution(kernel_size=3, output_depth=256, input_depth=128, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         Convolution(kernel_size=3, output_depth=256, input_depth=256, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         Convolution(kernel_size=3, output_depth=256, input_depth=256, batch_size=FLAGS.batch_size,
                       act='relu', stride_size=1, pad='SAME'),
         MaxPool(),

         Convolution(kernel_size=3, output_depth=512, input_depth=256, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         Convolution(kernel_size=3, output_depth=512, input_depth=512, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         Convolution(kernel_size=3, output_depth=512, input_depth=512, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         MaxPool(),

         Convolution(kernel_size=3, output_depth=512, input_depth=512, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         Convolution(kernel_size=3, output_depth=512, input_depth=512, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         Convolution(kernel_size=3, output_depth=512, input_depth=512, batch_size=FLAGS.batch_size,
                     act='relu', stride_size=1, pad='SAME'),
         MaxPool(),
         Convolution(kernel_size=7, output_depth=4096, stride_size=1, act='relu', pad='VALID'),
         Convolution(kernel_size=1, output_depth=4096, stride_size=1, act='relu', pad='VALID'),
         Convolution(kernel_size=1, output_depth=1000, stride_size=1, final = True, pad='VALID'),
         ])

def train():
    file_list = glob('./test_data/tmp_image/' + "*.jpg")
    gt = open('./test_data/val.txt','r').readlines()
    gt_list = []
    gt_num = []

    for i in range(len(gt)):
        tmp = gt[i].split()
        gt_list.append(tmp[0][:-5])
        gt_num.append(int(tmp[1]))
    img = []
    img_name = []
    gt_real = []
    for i in range(len(file_list)):
        if img == []:
            img = utils_vgg.load_image(file_list[i])
            img = np.expand_dims(img,0)
            # img_name.append(file_list[i].split('/')[-1]) # ubuntu
            img_name.append(file_list[i].split('\\')[-1])  # windows
            gt_real.append(gt_num[np.where(np.array(gt_list[:])==img_name[i][:-4])[0][0]])
        else:
            tmp = np.expand_dims(utils_vgg.load_image(file_list[i]),0)
            if tmp.shape[1:]==(224,224,3):
                img = np.concatenate([img,tmp],0)
                # img_name.append(file_list[i].split('/')[-1]) # ubuntu
                img_name.append(file_list[i].split('\\')[-1])  # windows
                gt_real.append(gt_num[np.where(np.array(gt_list[:])==img_name[i][:-4])[0][0]])
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

            y_ = tf.placeholder(tf.float32, shape=[None, 1000])
            phase = tf.placeholder(tf.bool, name='phase')
        with tf.variable_scope('model'):
            net = nn()
            inp = tf.reshape(x, [FLAGS.batch_size, 224, 224, 3])
            rgb_scaled = inp * 255.0
            VGG_MEAN = [103.939, 116.779, 123.68]
            # Convert RGB to BGR
            red, green, blue = tf.split(num_or_size_splits=3, value=rgb_scaled, axis=3)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat(values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
            assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
            op = net.forward(bgr)
            y = tf.reshape(op, [FLAGS.batch_size, 1000])

            soft = tf.nn.softmax(y)
        with tf.variable_scope('relevance'):
            if FLAGS.relevance:
                # q = tf.ones_like(soft)
                # one_hot = q*y_
                # kk = tf.one_hot(tf.argmax(tf.nn.softmax(y),-1),1000)
                mm = y_ # gt
                # mm = kk # pred
                if FLAGS.relevance_method == 'RAP':
                    LRP = []

                    RAP_pos, RAP_neg = net.RAP(y * mm, y * mm)
                    relevance_layerwise = []
                    relevance_layerwise_pos = []
                    relevance_layerwise_neg = []
                    R_p = y * mm
                    R_n = y * mm
                    for layer in net.modules[::-1]:
                        R_p, R_n = net.RAP_layerwise(layer, R_p, R_n)
                        relevance_layerwise_pos.append(R_p)
                        relevance_layerwise_neg.append(R_n)
                else:
                    RAP_pos = []
                    RAP_neg = []

                    LRP = net.RAP(y * mm, FLAGS.relevance_method)

                    relevance_layerwise = []
                    relevance_layerwise_pos = []
                    relevance_layerwise_neg = []
                    R = y * mm
                    for layer in net.modules[::-1]:
                        R = net.lrp_layerwise(layer, R, FLAGS.relevance_method)
                        relevance_layerwise.append(R)
            else:
                LRP = []
                relevance_layerwise = []
                relevance_layerwise_pos = []
                relevance_layerwise_neg = []
                RAP_pos = []
                RAP_neg = []
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

        # tf.global_variables_initializer().run()

        (x_test, y_test) = (img,gt_real)
        y_test_one_hot = tf.one_hot(y_test, 1000)
        for i in range(int(len(file_list)/FLAGS.batch_size)):
            d = next_batch(FLAGS.batch_size, x_test, y_test_one_hot.eval())

            test_inp = {x: d[0], y_: d[1], phase: False}
            # pdb.set_trace()
            summary, acc, relevance_test, RAP_p, RAP_n, op2, soft2, rel_layer, rel_layer_rap_p, rel_layer_rap_n = sess.run([merged, accuracy, LRP, RAP_pos, RAP_neg, y, soft, relevance_layerwise, relevance_layerwise_pos, relevance_layerwise_neg],
                                                               feed_dict=test_inp)
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
            ak = 0
            for m in range(FLAGS.batch_size):
                ax = np.argmax(soft2[m,:])
                print (op2[m,ax], soft2[m,ax], ax, d[1][m,ax])
                ak = ak+op2[m,ax]
                utils_vgg.print_prob(soft2[m], './synset.txt')

            if FLAGS.relevance_method == 'RAP':
                vis = RAP_p + RAP_n
                print([np.sum(rel) for rel in rel_layer_rap_p])
                print([np.sum(rel) for rel in rel_layer_rap_n])
            else:
                vis = relevance_test
                print([np.sum(rel) for rel in rel_layer])
            if FLAGS.relevance:
                # pdb.set_trace()
                # plot test images with relevances overlaid
                images = d[0].reshape([FLAGS.batch_size, 224, 224, 3])
                # images = (images + 1)/2.0
                plot_relevances(vis.reshape([FLAGS.batch_size, 224, 224, 3]),
                                images, test_writer)

        train_writer.close()
        test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()


