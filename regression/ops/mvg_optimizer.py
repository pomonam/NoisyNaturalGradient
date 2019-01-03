from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from regression.ops.ng_optimizer import NGOptimizer

import tensorflow as tf


class MVGOptimizer(NGOptimizer):
    def __init__(self, shape, N, lam, ita, alpha, beta, damp,
                 w_name, mu=None, fisher_u=None, fisher_v=None):
        super(MVGOptimizer, self).__init__(shape, N, lam, alpha, beta, w_name)
        self._init_mu(mu)
        self._init_fisher_u(fisher_u)
        self._init_fisher_v(fisher_v)
        self.ita = ita
        self.damp = damp
        self._init_bias_corr()
        self._init_momentum()

    def _init_mu(self, mu):
        if mu is None:
            self._mu = tf.get_variable(
                self.w_name + '_mean', shape=self.shape,
                initializer=tf.contrib.layers.xavier_initializer())
        else:
            self._mu = mu

    def _init_fisher_u(self, fisher_u):
        tmp = 0.1 * tf.random_normal([self.shape[0], self.shape[0]])
        if fisher_u is None:
            self._fisher_u = tf.get_variable(
                self.w_name + '_fisher_u',
                initializer=tf.eye(self.shape[0]) + tf.matmul(tmp, tf.transpose(tmp)))
        else:
            self._fisher_u = fisher_u

    def _init_fisher_v(self, fisher_v):
        tmp = 0.1 * tf.random_normal([self.shape[1], self.shape[1]])
        if fisher_v is None:
            self._fisher_v = tf.get_variable(
                self.w_name + '_fisher_v',
                initializer=tf.eye(self.shape[1]) + tf.matmul(tmp, tf.transpose(tmp)))
        else:
            self._fisher_v = fisher_v

    def _init_bias_corr(self):
        self._bias_corr = tf.get_variable(self.w_name + 'bias_corr', initializer=0., trainable=False)

    def _init_momentum(self):
        self._momentum = tf.get_variable(
                self.w_name + '_momentum', shape=self.shape,
                initializer=tf.constant_initializer(0.), trainable=False)

    @property
    def mu(self):
        return self._mu

    @property
    def u_damp(self):
        pi = tf.sqrt((tf.trace(self.fisher_v) / self.shape[1] + 1e-8)
                     / (tf.trace(self.fisher_u) / self.shape[0] + 1e-8))
        coeff = self.lam / (self.N * self.ita)
        u_tmp = tf.matrix_inverse(
            self.fisher_u + 1. / pi * (coeff + self.damp) ** 0.5 * tf.eye(self.shape[0]))
        return (self.lam / self.N) ** 0.5 * u_tmp

    @property
    def u(self):
        pi = tf.sqrt((tf.trace(self.fisher_v) / self.shape[1] + 1e-8)
                     / (tf.trace(self.fisher_u) / self.shape[0] + 1e-8))
        coeff = self.lam / (self.N * self.ita)
        u_tmp = tf.matrix_inverse(
            self.fisher_u + 1. / pi * coeff ** 0.5 * tf.eye(self.shape[0]))
        return (self.lam / self.N) ** 0.5 * u_tmp

    @property
    def v_damp(self):
        pi = tf.sqrt((tf.trace(self.fisher_v) / self.shape[1] + 1e-8)
                     / (tf.trace(self.fisher_u) / self.shape[0] + 1e-8))
        coeff = self.lam / (self.N * self.ita)
        v_tmp = tf.matrix_inverse(
            self.fisher_v + pi * (coeff + self.damp) ** 0.5 * tf.eye(self.shape[1]))
        return (self.lam / self.N) ** 0.5 * v_tmp

    @property
    def v(self):
        pi = tf.sqrt((tf.trace(self.fisher_v) / self.shape[1] + 1e-8)
                     / (tf.trace(self.fisher_u) / self.shape[0] + 1e-8))
        coeff = self.lam / (self.N * self.ita)
        v_tmp = tf.matrix_inverse(
            self.fisher_v + pi * coeff ** 0.5 * tf.eye(self.shape[1]))
        return (self.lam / self.N) ** 0.5 * v_tmp

    @property
    def fisher_u(self):
        return self._fisher_u

    @property
    def fisher_v(self):
        return self._fisher_v

    def update(self, w, w_grad, a, s_grad):
        """
        :param w: tensor with the same shape as self._mu and self._fisher
        :param log_py_xw: tensor. log likelihood.
        """
        coeff = self.lam / (self.N * self.ita)
        v_tmp = (self.N / self.lam) ** 0.5 * self.v_damp
        u_tmp = (self.N / self.lam) ** 0.5 * self.u_damp
        gradient = tf.reduce_mean(w_grad - coeff * w, 0)
        ratio = 0.1
        new_momentum = self._momentum * (1. - ratio) + gradient * ratio
        new_bias_corr = self._bias_corr * (1. - ratio) + ratio
        mu_grad = tf.matmul(u_tmp, tf.matmul(new_momentum / new_bias_corr, v_tmp))

        # tf.summary.scalar(self.w_name + '_mu_grad_abs', tf.reduce_mean(tf.abs(mu_grad)))
        # tf.summary.scalar(self.w_name + '_w_grad', tf.reduce_mean(w_grad))
        # tf.summary.scalar(self.w_name + '_w_grad_abs', tf.reduce_mean(tf.abs(w_grad)))
        # tf.summary.scalar(self.w_name + '_w', tf.reduce_mean(w))
        # tf.summary.scalar(self.w_name + '_w_abs', tf.reduce_mean(tf.abs(w)))
        infer1 = self._mu.assign(self._mu + self.alpha * mu_grad)

        a_t = tf.transpose(a, [0, 2, 1])
        fish_u = tf.reduce_mean(tf.matmul(a_t, a), [0]) / tf.to_float(tf.shape(a)[1])
        infer2 = self._fisher_u.assign((1. - self.beta) * self._fisher_u + self.beta * fish_u)

        s_grad_t = tf.transpose(s_grad, [0, 2, 1])
        fish_v = tf.reduce_mean(tf.matmul(s_grad_t, s_grad), [0]) / tf.to_float(tf.shape(s_grad)[1])
        infer3 = self._fisher_v.assign((1. - self.beta) * self._fisher_v + self.beta * fish_v)

        infer4 = self._momentum.assign(new_momentum)
        infer5 = self._bias_corr.assign(new_bias_corr)
        return [infer1, infer2, infer3, infer4, infer5]