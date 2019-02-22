# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import slim
from builders import frontend_builder
import tensorflow.contrib.layers as tf_layer
import numpy as np
import os, sys
import math

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

def AttentionRefinementModule(inputs, n_filters):

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    return net

def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net

# add by tmm
def Relu(inputdata, name=None):
    """

    :param name:
    :param inputdata:
    :return:
    """
    return tf.nn.relu(features=inputdata, name=name)


def _conv_stage(input_tensor, k_size,
                out_dims, name,stride=1, is_training=True, pad='SAME'):
    """
    将卷积和激活封装在一起
    :param input_tensor:
    :param k_size:
    :param out_dims:
    :param name:
    :param stride:
    :param pad:
    :return:
    """
    with tf.variable_scope(name):
        conv = conv2d(inputdata=input_tensor, out_channel=out_dims,
                      kernel_size=k_size, stride=stride,
                      use_bias=False, padding=pad, name='conv')
        bn = layerbn(inputdata=conv, is_training=is_training, name='bn')
        relu = Relu(inputdata=bn, name='relu')

    return relu


def layerbn(inputdata, is_training, name):
    """

    :param inputdata:
    :param is_training:
    :param name:
    :return:
    """

    def f1():
        """

        :return:
        """
        # print('batch_normalization: train phase')
        return tf_layer.batch_norm(
            inputdata, is_training=True,
            center=True, scale=True, updates_collections=None,
            scope=name, reuse=False)

    def f2():
        """

        :return:
        """
        # print('batch_normalization: test phase')
        return tf_layer.batch_norm(
            inputdata, is_training=False,
            center=True, scale=True, updates_collections=None,
            scope=name, reuse=True)

    output = tf.cond(is_training, f1, f2)

    return output

def dropout(inputdata, keep_prob, is_training=None, noise_shape=None, name=None):
    """

    :param name:
    :param inputdata:
    :param keep_prob:
    :param noise_shape:
    :return:
    """

    def f1():
        """

        :return:
        """
        # print('batch_normalization: train phase')
        return tf.nn.dropout(inputdata, keep_prob, noise_shape, name=name)

    def f2():
        """

        :return:
        """
        # print('batch_normalization: test phase')
        return inputdata

    output = tf.cond(is_training, f1, f2)
    # output = tf.nn.dropout(inputdata, keep_prob, noise_shape, name=name)

    return output

    # return tf.nn.dropout(inputdata, keep_prob=keep_prob, noise_shape=noise_shape, name=name)

def conv2d(inputdata, out_channel, kernel_size, padding='SAME',
           stride=1, w_init=None, b_init=None,
           split=1, use_bias=True, data_format='NHWC', name=None):
    """
    Packing the tensorflow conv2d function.
    :param name: op name
    :param inputdata: A 4D tensorflow tensor which ust have known number of channels, but can have other
    unknown dimensions.
    :param out_channel: number of output channel.
    :param kernel_size: int so only support square kernel convolution
    :param padding: 'VALID' or 'SAME'
    :param stride: int so only support square stride
    :param w_init: initializer for convolution weights
    :param b_init: initializer for bias
    :param split: split channels as used in Alexnet mainly group for GPU memory save.
    :param use_bias:  whether to use bias.
    :param data_format: default set to NHWC according tensorflow
    :return: tf.Tensor named ``output``
    """
    with tf.variable_scope(name):
        in_shape = inputdata.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0
        assert out_channel % split == 0

        padding = padding.upper()

        if isinstance(kernel_size, list):
            filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel / split, out_channel]
        else:
            filter_shape = [kernel_size, kernel_size] + [in_channel / split, out_channel]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        w = tf.get_variable('W', filter_shape, initializer=w_init)
        b = None

        if use_bias:
            b = tf.get_variable('b', [out_channel], initializer=b_init)

        if split == 1:
            conv = tf.nn.conv2d(inputdata, w, strides, padding, data_format=data_format)
        else:
            inputs = tf.split(inputdata, split, channel_axis)
            kernels = tf.split(w, split, 3)
            outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                       for i, k in zip(inputs, kernels)]
            conv = tf.concat(outputs, channel_axis)

        ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, name=name)

    return ret



def build_bisenet(inputs, num_classes, preset_model='BiSeNet', frontend="ResNet101", weight_decay=1e-5, is_training=True, pretrained_dir="models"):
    """
    Builds the BiSeNet model. 

    Arguments:
      inputs: The input tensor=
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      BiSeNet model
    """

    ### The spatial path
    ### The number of feature maps for each convolution is not specified in the paper
    ### It was chosen here to be equal to the number of feature maps of a classification
    ### model at each corresponding stage 
    spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[3, 3], strides=2)


    #the following added by tmm
    # conv stage 5_5
    #conv_5_5 = _conv_stage(input_tensor=spatial_net, k_size=1,
    #                            out_dims=128, is_training = is_training, name='conv5_5')  # 4 x 36 x 100 x 128

    # add message passing #
    # top to down #
    feature_list_old = []
    feature_list_new = []
    conv_5_5 = spatial_net
    for cnt in range(conv_5_5.get_shape().as_list()[1]):
        feature_list_old.append(tf.expand_dims(conv_5_5[:, cnt, :, :], axis=1))
    feature_list_new.append(tf.expand_dims(conv_5_5[:, 0, :, :], axis=1))

    w1 = tf.get_variable('W1', [1, 9, 128, 128],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
    with tf.variable_scope("convs_6_1"):
        conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[1])
        feature_list_new.append(conv_6_1)

    for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
        with tf.variable_scope("convs_6_1", reuse=True):
            conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[cnt])
            feature_list_new.append(conv_6_1)

    # down to top #
    feature_list_old = feature_list_new
    feature_list_new = []
    #length = int(CFG.TRAIN.IMG_HEIGHT / 8) - 1
    length = int(inputs.get_shape().as_list()[1] / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w2 = tf.get_variable('W2', [1, 9, 128, 128],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
    with tf.variable_scope("convs_6_2"):
        conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[length - 1])
        feature_list_new.append(conv_6_2)

    for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
        with tf.variable_scope("convs_6_2", reuse=True):
            conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[length - cnt])
            feature_list_new.append(conv_6_2)

    feature_list_new.reverse()

    processed_feature = tf.stack(feature_list_new, axis=1)
    processed_feature = tf.squeeze(processed_feature, axis=2)

    # left to right #

    feature_list_old = []
    feature_list_new = []
    for cnt in range(processed_feature.get_shape().as_list()[2]):
        feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
    feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

    w3 = tf.get_variable('W3', [9, 1, 128, 128],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
    with tf.variable_scope("convs_6_3"):
        conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[1])
        feature_list_new.append(conv_6_3)

    for cnt in range(2, processed_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_6_3", reuse=True):
            conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[cnt])
            feature_list_new.append(conv_6_3)

    # right to left #

    feature_list_old = feature_list_new
    feature_list_new = []
    #length = int(CFG.TRAIN.IMG_WIDTH / 8) - 1
    length = int(inputs.get_shape().as_list()[2] / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w4 = tf.get_variable('W4', [9, 1, 128, 128],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
    with tf.variable_scope("convs_6_4"):
        conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[length - 1])
        feature_list_new.append(conv_6_4)

    for cnt in range(2, processed_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_6_4", reuse=True):
            conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[length - cnt])
            feature_list_new.append(conv_6_4)

    feature_list_new.reverse()
    processed_feature = tf.stack(feature_list_new, axis=2)
    processed_feature = tf.squeeze(processed_feature, axis=3)
    #######################
    dropout_output = dropout(processed_feature, 0.9, is_training=is_training,
                                  name='dropout')  # 0.9 denotes the probability of being kept

    #conv_output = conv2d(inputdata=dropout_output, out_channel=5,
    #                          kernel_size=1, use_bias=True, name='conv_6')
    #ret['prob_output'] = tf.image.resize_images(conv_output, [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])
    #spatial_net = tf.image.resize_images(conv_output, [tf.shape(inputs)[1], tf.shape(inputs)[2]])
    spatial_net = dropout_output
    #end by tmm

    spatial_net = ConvBlock(spatial_net, n_filters=256, kernel_size=[3, 3], strides=2)

    ### Context path
    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)

    net_4 = AttentionRefinementModule(end_points['pool4'], n_filters=512)

    net_5 = AttentionRefinementModule(end_points['pool5'], n_filters=2048)

    global_channels = tf.reduce_mean(net_5, [1, 2], keep_dims=True)
    net_5_scaled = tf.multiply(global_channels, net_5)

    ### Combining the paths
    net_4 = Upsampling(net_4, scale=2)
    net_5_scaled = Upsampling(net_5_scaled, scale=4)

    context_net = tf.concat([net_4, net_5_scaled], axis=-1)

    net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters=num_classes)


    ### Final upscaling and finish
    net = Upsampling(net, scale=8)
    
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn

