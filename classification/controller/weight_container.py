from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def _compute_pi_tracenorm(left_cov, right_cov):
    left_norm = tf.trace(left_cov) * right_cov.shape.as_list()[0]
    right_norm = tf.trace(right_cov) * left_cov.shape.as_list()[0]
    return tf.sqrt(left_norm / right_norm)


class WeightContainer(object):
    def __init__(self, data_size, shape, layer_idx, coeff, eta):
        """ Initialize a class Weight Container.
        :param data_size: int
        :param shape: tuple(int,)
        :param layer_idx: int
        :param coeff: int
        :param eta: int
        """
        self._data_size = data_size
        self._shape = shape
        self._in_channels, self._out_channels = np.prod(self._shape[:-1]) + 1, self._shape[-1]
        self._idx = layer_idx
        self._coeff = coeff
        self._eta = eta

        # Initialize weight and bias.
        self._weight = tf.get_variable("weight_{}".format(str(self._idx)),
                                       shape=self._shape,
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=True)
        self._bias = tf.get_variable("bias_{}".format(str(self._idx)),
                                     shape=[self._out_channels],
                                     initializer=tf.constant_initializer(0.),
                                     trainable=True)

    def _combine_weight_bias(self):
        """ Combine weight and bias to sample the weight.
        :return: 2D Tensor of shape 'in_channels x out_channels'
        """
        weight = tf.reshape(self._weight, (self._in_channels - 1, self._out_channels))
        bias = tf.expand_dims(self._bias, 0)
        return tf.concat([weight, bias], 0)

    def params(self):
        """ Return 2-tuple of Tensors corresponding to weight and bias of this layer.
        Bias has shape 'out_channels'
        :return: Tuple(Weight, Bias)
        """
        return self._get_weight(), self._get_bias()

    def _get_weight(self):
        """ Return weight of the current weight container.
        :return: Tensor with self.shape.
        """
        return self._weight

    def _get_bias(self):
        """ Return bias of the current weight container.
        :return: Tensor with self.shape
        """
        return self._bias

    def sample_weight(self, particles):
        """ Sample weight from the variational posterior.
        :return: 2D Tensor with size 'batch_size x feature_size'
        """
        raise NotImplementedError()

    def update_weight(self, *hyper_params, **unused_inputs):
        """ Update the weight of variational posterior.
        :return: Tensor update operation.
        """
        raise NotImplementedError()

    def update_scale(self, fisher_block):
        return None


class FFGWeightContainer(WeightContainer):
    def __init__(self, data_size, shape, layer_idx, coeff, eta):
        """ Initialize Fully-Factorized Gaussian (FFG) Weight Container. """
        super(FFGWeightContainer, self).__init__(data_size, shape, layer_idx, coeff, eta)

        # Initialize the standard deviation.
        self._std = tf.get_variable("std_weight_{}".format(str(self._idx)),
                                    [self._in_channels, self._out_channels],
                                    initializer=tf.constant_initializer(1e-5),
                                    trainable=False)

    def sample_weight(self, particles):
        """ Sample weight from the variational posterior.
        In the paper, w ~ N(u, (lambda / num_data) * inv(diag(f + gamma_in)))
        :return: 2D Tensor with size 'batch_size x feature_size'
        """
        # Get homogeneous weight (weight + bias); Act as a mean.
        homo_weight = self._combine_weight_bias()
        multi_weight = tf.tile(tf.expand_dims(homo_weight, 0), [particles, 1, 1])
        noise = tf.random_normal(shape=tf.shape(multi_weight))
        return multi_weight + self._std * noise

    def update_weight(self, fisher_block):
        """ Update the weight of variational posterior.
        :return: Update operation.
        """
        est_f = fisher_block._factor.get_cov()
        damped_est_f = est_f + tf.divide(self._coeff, self._eta)
        variance = 1 / damped_est_f
        weight_update_op = self._std.assign(tf.sqrt(self._coeff * variance))
        return weight_update_op


class MVGWeightContainer(WeightContainer):
    def __init__(self, data_size, shape, layer_idx, coeff, eta):
        """ Initialize Matrix Variate Gaussian (MVG) Weight Container. """
        super(MVGWeightContainer, self).__init__(data_size, shape, layer_idx, coeff, eta)

        # U and V refers to covariance of the rows and columns accordingly.
        self._u = tf.get_variable("u_weight_{}".format(str(self._idx)),
                                  initializer=1e-3 * tf.eye(self._in_channels),
                                  trainable=False)
        self._v = tf.get_variable("v_weight_{}".format(str(self._idx)),
                                  initializer=1e-3 * tf.eye(self._out_channels),
                                  trainable=False)

    def sample_weight(self, particles):
        """ Sample weight from the variational posterior.
        In the paper, W ~ MN(M, (lambda / num_data) * inv(A^gamma_in), inv(S^gamma_in))
        :return: 2D Tensor with size 'batch_size x feature_size'
        """
        # Get homogeneous weight (weight + bias); Act as a mean.
        homo_weight = self._combine_weight_bias()
        multi_weight = tf.tile(tf.expand_dims(homo_weight, 0), [particles, 1, 1])
        noise = tf.random_normal(shape=tf.shape(multi_weight))
        u_c = tf.tile(tf.expand_dims(self._u, 0), [particles, 1, 1])
        v_c = tf.tile(tf.expand_dims(self._v, 0), [particles, 1, 1])

        return multi_weight + tf.matmul(u_c, tf.matmul(noise, v_c, transpose_b=True))

    def update_weight(self, fisher_block):
        """ Update the weight of variational posterior.
        :return: Update operations.
        """
        input_factor = fisher_block._input_factor
        output_factor = fisher_block._output_factor
        input_cov, output_cov = input_factor.get_cov(), output_factor.get_cov()

        # Configure damping parameters.
        pi = _compute_pi_tracenorm(input_cov, output_cov)
        coeff = (self._coeff / fisher_block._renorm_coeff) ** 0.5
        damping = coeff / (self._eta ** 0.5)

        ue, uv = tf.self_adjoint_eig(
            input_cov / pi + damping * tf.eye(self._u.shape.as_list()[0]))
        ve, vv = tf.self_adjoint_eig(
            output_cov * pi + damping * tf.eye(self._v.shape.as_list()[0]))

        ue = coeff / tf.maximum(ue, damping)
        new_u = uv * tf.sqrt(ue)

        ve = coeff / tf.maximum(ve, damping)
        new_v = vv * tf.sqrt(ve)

        weight_update_ops = [self._u.assign(new_u), self._v.assign(new_v)]

        return tf.group(*weight_update_ops)


class EMVGWeightContainer(WeightContainer):
    def __init__(self, data_size, shape, layer_idx, coeff, eta):
        """ Initialize Eigenvalue Corrected Matrix Variate Gaussian (EMVG) Weight Container. """
        super(EMVGWeightContainer, self).__init__(data_size, shape, layer_idx, coeff, eta)

        self._u = tf.get_variable("u_weight_{}".format(str(self._idx)),
                                  initializer=1e-3 * tf.eye(self._in_channels),
                                  trainable=False)
        self._v = tf.get_variable("v_weight_{}".format(str(self._idx)),
                                  initializer=1e-3 * tf.eye(self._out_channels),
                                  trainable=False)
        self._s = tf.get_variable("s_weight_{}".format(str(self._idx)),
                                  initializer=1e-5 * tf.ones([self._in_channels, self._out_channels]),
                                  trainable=False)

    def sample_weight(self, particles):
        """ Sample weight from the variational posterior.
        In the paper, W ~ MN(M, (lambda / num_data) * inv(A^gamma_in), inv(S^gamma_in))
        :return: 2D Tensor with size 'batch_size x feature_size'
        """
        # Get homogeneous weight (weight + bias); Act as a mean.
        homo_weight = self._combine_weight_bias()
        multi_weight = tf.tile(tf.expand_dims(homo_weight, 0), [particles, 1, 1])
        noise = tf.random_normal(shape=tf.shape(multi_weight))
        u_c = tf.tile(tf.expand_dims(self._u, 0), [particles, 1, 1])
        v_c = tf.tile(tf.expand_dims(self._v, 0), [particles, 1, 1])
        s_c = tf.tile(tf.expand_dims(self._s, 0), [particles, 1, 1])

        reshaped_out = tf.multiply(noise, s_c)
        return multi_weight + tf.matmul(u_c, tf.matmul(reshaped_out, v_c, transpose_b=True))

    def update_weight(self, fisher_block):
        """ Update the weight of variational posterior.
        :return: Update operations.
        """
        left_eigen_basis = fisher_block._scale_factor._input_factor_eigen_basis
        right_eigen_basis = fisher_block._scale_factor._output_factor_eigen_basis
        weight_update_ops = [self._u.assign(left_eigen_basis),
                             self._v.assign(right_eigen_basis)]

        return tf.group(*weight_update_ops)

    def update_scale(self, fisher_block):
        """ Update the scale of variational posterior.
        :return: Update operations
        """
        scale_factor = fisher_block._scale_factor
        scale_cov = scale_factor.get_cov()
        # Add by intrinsic damping term.
        damped_scale_cov = scale_cov + tf.divide(self._coeff, self._eta)
        damped_inverse_scale_cov = self._coeff / damped_scale_cov
        scale_update_op = [self._s.assign(tf.sqrt(damped_inverse_scale_cov))]
        return tf.group(*scale_update_op)
