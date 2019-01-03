from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from regression.misc.collections import get_collection
from regression.misc.kl_utils import *
from regression.controller.distributions import *

import tensorflow as tf
import zhusuan as zs


def _append_homog(tensor):
    rank = len(tensor.shape.as_list())
    shape = tf.concat([tf.shape(tensor)[:-1], [1]], axis=0)
    ones = tf.ones(shape, dtype=tensor.dtype)
    return tf.concat([tensor, ones], axis=rank - 1)


class Layer(object):
    """ The :class: Layer is the base class for various kind of layers in a
    Bayesian neural network setting.

    :param n_in: int.
    :param n_out: int.
    :param w_name: str. name of StochasticTensor weight.
    """

    def __init__(self, n_in, n_out, w_name):
        self._n_in = n_in
        self._n_out = n_out
        self._w_name = w_name
        super(Layer, self).__init__()

    @property
    def n_in(self):
        return self._n_in

    @property
    def n_out(self):
        return self._n_out

    @property
    def w_name(self):
        return self._w_name

    def forward(self, inputs):
        return NotImplementedError()


class NormalPrior(Layer):
    """ The :class: `NormalPrior` is the inherited class from :class: `Layer`. Representing
    layers with normal distribution priors.

    :param n_in: int.
    :param n_out: int.
    :param w_name: str. name of StochasticTensor weight.
    :param params: dict. Cotaining 'p_mean': tensor, 'p_logstd': tensor. If not provided, mean and
    logstd will be set to 0.
    """

    def __init__(self, n_in, n_out, w_name, params=None):
        super(NormalPrior, self).__init__(n_in, n_out, w_name)
        if params is None or (isinstance(params, dict) and len(params) == 0):
            mean = 0.
            logstd = 0.
        elif isinstance(params, dict) and 'mean' in params and 'logstd' in params:
            mean = params['p_mean']
            logstd = params['p_logstd']
        else:
            raise ValueError('Not an appropriate param')
        self._p_mean = mean * tf.ones([self.n_in + 1, self.n_out])
        self._p_logstd = logstd * tf.ones([self.n_in + 1, self.n_out])

    @property
    def p_mean(self):
        return self._p_mean

    @property
    def p_logstd(self):
        return self._p_logstd

    def _p_samples(self, n_particles):
        """ Generate samples from a gaussian distribution.
        :param n_particles: tensor. Number of samples to be generated.
        :return: tensor of shape [n_particles, n_in+1, n_out]
        """
        self._pws = zs.Normal(self.w_name,
                              mean=self.p_mean,
                              logstd=self.p_logstd,
                              group_ndims=2, n_samples=n_particles)

    def pws(self, n_particles):
        if not hasattr(self, '_pws'):
            self._p_samples(n_particles)
        assert_same_num = tf.assert_equal(
            n_particles, tf.shape(self._pws)[0],
            message='n_particles must be the same with the previous one')
        with tf.control_dependencies([assert_same_num]):
            return self._pws


class FeedForward(NormalPrior):
    """ The :class: `NormalPrior` is the inherited class from :class: `Layer`. Representing
    layers with normal distribution priors.

    :param n_in: int.
    :param n_out: int.
    :param w_name: str. name of StochasticTensor weight.
    :param params: dict. Cotaining 'p_mean': tensor, 'p_logstd': tensor. If not provided, mean and
    logstd will be set to 0.
    """
    def __init__(self, n_in, n_out, w_name, params=None):
        super(FeedForward, self).__init__(n_in, n_out, w_name, params)

    def forward(self, inputs):
        inputs = tf.concat(
            [inputs,
             tf.ones(tf.concat([tf.shape(inputs)[:-1], [1]], axis=0))], axis=-1)
        pws = self.pws(tf.shape(inputs)[0])
        return tf.matmul(inputs, pws)

    def _q_samples(self, n_particles):
        raise NotImplementedError

    def qws(self, n_particles):
        if not hasattr(self, '_qws'):
            self._q_samples(n_particles)
        assert_same_num = tf.assert_equal(
            n_particles, tf.shape(self._qws)[0],
            message='n_particles must be the same with the previous one')
        with tf.control_dependencies([assert_same_num]):
            return self._qws

    @property
    def kl_appro(self):
        return self._qws.log_prob(self._qws) - self._pws.log_prob(self._pws)


class MVGLayer(FeedForward):
    """ The :class: `MVGLayer` is the inherited class from :class: `Layer`.
    """
    def __init__(self, n_in, n_out, w_name, params=None):
        super(MVGLayer, self).__init__(n_in, n_out, w_name, params)

    def _q_samples(self, n_particles):
        self._q_mean = get_collection(self.w_name + '_q_mean')
        self._q_u = get_collection(self.w_name + '_q_u')
        self._q_v = get_collection(self.w_name + '_q_v')
        self._qws = MatrixVariateNormal(self.w_name,
                                        mean=self._q_mean,
                                        u=self._q_u,
                                        v=self._q_v,
                                        n_samples=n_particles,
                                        group_event_ndims=0)

    @property
    def q_u(self):
        return self._q_u

    @property
    def q_mean(self):
        return self._q_mean

    @property
    def q_v(self):
        return self._q_v

    @property
    def kl_exact(self):
        return mvg_mf((self.q_mean, self.q_u, self.q_v),
                      (self.p_mean, self.p_logstd))

    # def forward(self, inputs):
    #     inputs = tf.concat(
    #         [inputs,
    #          tf.ones(tf.concat([tf.shape(inputs)[:-1], [1]], axis=0))], axis=-1)
    #     qws = self.qws(tf.shape(inputs)[0])
    #     return tf.matmul(inputs, qws)


class EMVGLayer(FeedForward):
    """
    The :class: `EMVGLayer` is the inherited class from :class: `Layer`.
    """
    def __init__(self, n_in, n_out, w_name, params=None):
        super(EMVGLayer, self).__init__(n_in, n_out, w_name, params)

    def _q_samples(self, n_particles):
        self._q_mean = get_collection(self.w_name + '_q_mean')
        self._q_u_b = get_collection(self.w_name + '_q_u_b')
        self._q_v_b = get_collection(self.w_name + '_q_v_b')
        self._q_r = get_collection(self.w_name + '_q_r')
        self._qws = EigenMatrixVariateNormal(self.w_name,
                                             mean=self._q_mean,
                                             u_b=self._q_u_b,
                                             v_b=self._q_v_b,
                                             r=self._q_r,
                                             n_samples=n_particles,
                                             group_event_ndims=0)

    @property
    def q_u_b(self):
        return self._q_u_b

    @property
    def q_mean(self):
        return self._q_mean

    @property
    def q_v_b(self):
        return self._q_v_b

    @property
    def q_r(self):
        return self._q_r

    @property
    def kl_exact(self):
        return emvg_mf((self.q_mean, self.q_u_b, self.q_v_b, self.q_r),
                       (self.p_mean, self.p_logstd))

    # def forward(self, inputs):
    #     inputs = tf.concat(
    #         [inputs,
    #          tf.ones(tf.concat([tf.shape(inputs)[:-1], [1]], axis=0))], axis=-1)
    #     qws = self.qws(tf.shape(inputs)[0])
    #     return tf.matmul(inputs, qws)
