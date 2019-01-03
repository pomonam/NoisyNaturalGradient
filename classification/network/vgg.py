from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from misc.registry import register_model
from classification.misc.layers import conv2d, dense

import tensorflow as tf
import numpy as np


def vgg(inputs, controller, is_training, add_batch_norm, layer_collection, particles, num_blocks):
    def vgg_block(inputs, layers, out_channel):
        """ Construct a VGG block.
        :return: Tensor, float
        """
        l2_loss = 0.
        for l in range(layers):
            in_channel = inputs.shape.as_list()[-1]
            container, idx = controller.register_conv_layer(3, in_channel, out_channel)
            sampled_weight = controller.get_weight(idx)
            pre, act = conv2d(inputs, sampled_weight, (3, 3, in_channel, out_channel),
                              add_batch_norm, is_training, particles, padding="SAME")
            layer_collection.register_conv2d(controller.get_params(idx), (1, 1, 1, 1), "SAME", inputs, pre)
            inputs = act
            l2_loss += 0.5 * tf.reduce_sum(sampled_weight ** 2)
        outputs = tf.layers.max_pooling2d(inputs, 2, 2, "SAME")
        return outputs, l2_loss

    inputs = tf.tile(inputs, [particles, 1, 1, 1])
    # block 1
    layer1, l2_loss1 = vgg_block(inputs, num_blocks[0], 32)
    # block 2
    layer2, l2_loss2 = vgg_block(layer1, num_blocks[1], 64)
    # block 3
    layer3, l2_loss3 = vgg_block(layer2, num_blocks[2], 128)
    # block 4
    layer4, l2_loss4 = vgg_block(layer3, num_blocks[3], 256)
    # block 5
    layer5, l2_loss5 = vgg_block(layer4, num_blocks[4], 256)

    # l2_loss; Add to the loss.
    l2_loss = l2_loss1 + l2_loss2 + l2_loss3 + l2_loss4 + l2_loss5

    flat = tf.reshape(layer5, shape=[-1, int(np.prod(layer5.shape[1:]))])
    container, idx = controller.register_fc_layer(256, 10)
    weights = container.sample_weight(particles)
    l2_loss += 0.5 * tf.reduce_sum(weights ** 2)
    logits, _ = dense(flat, weights, add_batch_norm, is_training, particles)
    layer_collection.register_fully_connected(container.params(), flat, logits)
    layer_collection.register_categorical_predictive_distribution(logits, name="logits")

    return logits, l2_loss


@register_model("vgg11")
def vgg11(inputs, sampler, is_training, batch_norm, layer_collection, particles):
    return vgg(inputs, sampler, is_training, batch_norm, layer_collection, particles, [1, 1, 2, 2, 2])


@register_model("vgg13")
def vgg13(inputs, sampler, is_training, batch_norm, layer_collection, particles):
    return vgg(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 2, 2, 2, 2])


@register_model("vgg16")
def vgg16(inputs, sampler, is_training, batch_norm, layer_collection, particles):
    return vgg(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 2, 3, 3, 3])


@register_model("vgg19")
def vgg19(inputs, sampler, is_training, batch_norm, layer_collection, particles):
    return vgg(inputs, sampler, is_training, batch_norm, layer_collection, particles, [2, 2, 4, 4, 4])
