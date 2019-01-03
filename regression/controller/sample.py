from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from regression.misc.kl_utils import *

import tensorflow as tf
import zhusuan as zs
import numpy as np


class OutSample(object):
    """ The :class:`OutSample` is the base class for various sampling methods
    based on the output of the forwarding process.
    """
    def __init__(self, params=None, stochastic_names=[], task=None):
        self._stochastic_names = stochastic_names
        self._task = task

        if isinstance(params, dict):
            for k, v in params.items():
                setattr(self, k, v)

    @property
    def stochastic_names(self):
        return self._stochastic_names

    @property
    def task(self):
        return self._task

    def p_sample(self, h):
        """
        Sample from prior distribution.

        :param h: Tensor of shape [n_particles, batch_size, n].
            Representing the output of former forwarding process.

        :return: A tensor of shape [n_particles, batch_size].
        """
        return NotImplementedError

    @staticmethod
    def q_sample(h):
        """
        Sample from variational distribution.

        :param h: Tensor of shape [n_particles, batch_size, n].
            Representing the output of former forwarding process.

        :return: A tensor of shape [n_particles, batch_size].
        """
        return NotImplementedError


class NormalOutSample(OutSample):
    """
    The :class:`NormalOutSample` is the inherited class of :class:`OutSample`.
    Representing sampling outputs from a Normal distribution.
    """
    def __init__(self, params=None):
        super(NormalOutSample, self).__init__(params, ['y_prec', 'y'], 'regression')

    def _p_samples(self, n_particles):
        if not hasattr(self, '_p_alpha'):
            self._p_alpha = 6.
        if not hasattr(self, '_p_beta'):
            self._p_beta = 6.
        self._p_prec = zs.Gamma('y_prec', self._p_alpha, self._p_beta,
                                n_samples=n_particles)

    @property
    def p_alpha(self):
        return self._p_alpha

    @property
    def p_beta(self):
        return self._p_beta

    def ps(self, n_particles):
        if not hasattr(self, '_p_prec'):
            self._p_samples(n_particles)
        assert_same_num = tf.assert_equal(
            n_particles, tf.shape(self._p_prec)[0],
            message='n_particles must be the same with the previous one')
        with tf.control_dependencies([assert_same_num]):
            return self._p_prec

    def _q_samples(self, n_particles):
        """ Sampling precision for output from variational posterior.
        :param n_particles: tensor. Number of samples.
        :return: tensor of shape [n_particles, 1]. Samples of precision.
        """
        prec_logalpha = tf.get_variable(
            'q_prec_logalpha', shape=[1],
            initializer=tf.constant_initializer(np.log(6.)))
        prec_logbeta = tf.get_variable(
            'q_prec_logbeta', shape=[1],
            initializer=tf.constant_initializer(np.log(6.)))
        self._q_alpha = tf.exp(prec_logalpha)
        self._q_beta = tf.exp(prec_logbeta)
        self._q_prec = zs.Gamma(
            'y_prec', self._q_alpha, self._q_beta,
            n_samples=n_particles, group_ndims=1)

    def qs(self, n_particles):
        if not hasattr(self, '_q_prec'):
            self._q_samples(n_particles)
        assert_same_num = tf.assert_equal(
            n_particles, tf.shape(self._q_prec)[0],
            message='n_particles must be the same with the previous one')
        with tf.control_dependencies([assert_same_num]):
            return self._q_prec

    @property
    def q_alpha(self):
        return self._q_alpha

    @property
    def q_beta(self):
        return self._q_beta

    def forward(self, mean):
        mean = tf.squeeze(mean, [-1])
        prec = self.ps(tf.shape(mean)[0])
        prec = tf.stop_gradient(prec)
        y_logstd = -0.5 * tf.log(prec)
        y = zs.Normal('y', mean, logstd=y_logstd * tf.ones_like(mean))
        return y

    @property
    def kl_exact(self):
        return gamma_gamma((self.q_alpha, self.q_beta),
                           (self.p_alpha, self.p_beta))
