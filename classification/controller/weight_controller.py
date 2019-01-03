from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from classification.controller.weight_container import *

import tensorflow as tf


_VARIATION_TYPE = {
    "diagonal": FFGWeightContainer,
    "kron": MVGWeightContainer,
    "eigen": EMVGWeightContainer,
}


class WeightController(object):
    def __init__(self, data_size, config, particles):
        """ Initialize class WeightController.
        :param data_size: int
            The size of the dataset.
        :param config: dict{String:Object}
            Configuration for optimizer.
        :param particles: int
            The number of particles.
        """
        # Ensure that variation_type has the correct value.
        self._data_size = data_size
        self._config = config
        self._variation_type = self._config.fisher_approx

        # Set up the hyper-parameters.
        self._coeff = self._config.kl / float(self._data_size)
        self._eta = config.eta
        self._particles = particles

        # Controls the weights of all layers.
        # {index (int) : WeightContainer}
        self._controller = list()
        self._num_layers = 0

    def register_conv_layer(self, kernel_size, in_channel, out_channel):
        """ Register a conv2d layer.
        :param kernel_size: int
        :param in_channel: int
        :param out_channel: int
        :return: WeightContainer
        """
        cur_idx = self._num_layers
        container = _VARIATION_TYPE[self._variation_type](data_size=self._data_size,
                                                          shape=[kernel_size, kernel_size, in_channel, out_channel],
                                                          layer_idx=cur_idx,
                                                          coeff=self._coeff,
                                                          eta=self._eta)
        # Automatically keep track of the number of layers.
        self._controller.append(container)
        self._num_layers += 1
        return container, cur_idx

    def register_fc_layer(self, input_size, output_size):
        """ Register a fully-connected layer.
        :param input_size: int
        :param output_size: int
        :return: WeightContainer
        """
        cur_idx = self._num_layers
        container = _VARIATION_TYPE[self._variation_type](data_size=self._data_size,
                                                          shape=[input_size, output_size],
                                                          layer_idx=cur_idx,
                                                          coeff=self._coeff,
                                                          eta=self._eta)
        # Automatically keep track of the number of layers.
        self._controller.append(container)
        self._num_layers += 1
        return container, cur_idx

    def get_weight(self, layer_idx):
        """ Return the current weight of the layer with the given layer_idx.
        :param layer_idx: int
        :return: 2D Tensor with size 'batch_size x feature_size'
        """
        if layer_idx >= len(self._controller):
            raise IndexError("Cannot find the specified layer_idx, {}.".format(str(layer_idx)))
        return self._controller[layer_idx].sample_weight(self._particles)

    def get_params(self, layer_idx):
        """ Return the params for each layer.
        :param layer_idx: int
        :return: Tuple(Weight, Bias)
        """
        if layer_idx >= len(self._controller):
            raise IndexError("Cannot find the specified layer_idx, {}.".format(str(layer_idx)))
        return self._controller[layer_idx].params()

    def update_weights(self, blocks):
        """ Update the weight of the layer with the given_idx."""
        # Get all the layers.
        update_ops = [cont.update_weight(blk) for cont, blk in zip(self._controller, blocks)]
        return tf.group(*update_ops)

    def update_scales(self, blocks):
        """ Update the scale of the layer with the given_idx."""
        update_ops = [cont.update_scale(blk) for cont, blk in zip(self._controller, blocks)]
        return tf.group(*update_ops)
