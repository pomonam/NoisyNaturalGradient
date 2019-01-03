from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from misc.registry import register_model
from regression.misc.collections import add_to_collection
from regression.ops import emvg_optimizer, mvg_optimizer

import tensorflow as tf


def ffn(layer_type, input_size, num_hidden, num_data, kl_factor, ita, alpha, beta, damp, omega=None):
    valid_layer_type = ["emvg", "mvg"]
    layer_type = layer_type.lower()
    if layer_type not in valid_layer_type:
        raise ValueError("Unavailable layer type %s" % layer_type)

    init_ops = []
    layers = []

    if layer_type == "emvg":
        layer1 = emvg_optimizer.EMVGOptimizer([input_size + 1, num_hidden],
                                              num_data, kl_factor, ita, alpha,
                                              beta, damp, "w0", omega=omega)
        add_to_collection('w0' + '_q_mean', layer1.mu)
        add_to_collection('w0' + '_q_u_b', layer1.u_b)
        add_to_collection('w0' + '_q_v_b', layer1.v_b)
        add_to_collection('w0' + '_q_r', layer1.r)
        tf.summary.scalar('w0' + '_q_mean', tf.reduce_mean(layer1.mu))
        init_ops.append(layer1.init_r_kfac())

        layer2 = emvg_optimizer.EMVGOptimizer([num_hidden + 1, 1],
                                              num_data, kl_factor, ita, alpha,
                                              beta, damp, "w1", omega=omega)
        add_to_collection('w1' + '_q_mean', layer2.mu)
        add_to_collection('w1' + '_q_u_b', layer2.u_b)
        add_to_collection('w1' + '_q_v_b', layer2.v_b)
        add_to_collection('w1' + '_q_r', layer2.r)
        tf.summary.scalar('w1' + '_q_mean', tf.reduce_mean(layer2.mu))
        init_ops.append(layer2.init_r_kfac())

    elif layer_type == "mvg":
        layer1 = mvg_optimizer.MVGOptimizer([input_size + 1, num_hidden],
                                            num_data, kl_factor, ita, alpha,
                                            beta, damp, "w0")
        add_to_collection('w0' + '_q_mean', layer1.mu)
        add_to_collection('w0' + '_q_u', layer1.u)
        add_to_collection('w0' + '_q_v', layer1.v)

        layer2 = mvg_optimizer.MVGOptimizer([num_hidden + 1, 1],
                                            num_data, kl_factor, ita, alpha,
                                            beta, damp, "w1")
        add_to_collection('w1' + '_q_mean', layer2.mu)
        add_to_collection('w1' + '_q_u', layer2.u)
        add_to_collection('w1' + '_q_v', layer2.v)

    else:
        raise NotImplementedError()

    layers.append(layer1)
    layers.append(layer2)

    return layers, init_ops, num_hidden


@register_model("ffn50")
def ffn50(layer_type, input_size, num_data, kl_factor, ita, alpha, beta, damp, omega=None):
    return ffn(layer_type, input_size, 50, num_data, kl_factor, ita, alpha, beta, damp, omega=omega)


@register_model("ffn100")
def ffn50(layer_type, input_size, num_data, kl_factor, ita, alpha, beta, damp, omega=None):
    return ffn(layer_type, input_size, 100, num_data, kl_factor, ita, alpha, beta, damp, omega=omega)
