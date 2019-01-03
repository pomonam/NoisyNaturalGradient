from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from regression.ops.ng_optimizer import NGOptimizer

import tensorflow as tf


class EMVGOptimizer(NGOptimizer):
    def __init__(self, shape, N, lam, ita, alpha, beta, damp,
                 w_name, mu=None, fisher_u=None, fisher_v=None, r=None, omega=None):
        super(EMVGOptimizer, self).__init__(shape, N, lam, alpha, beta, w_name)
        self._init_mu(mu)
        self._init_fisher_u(fisher_u)
        self._init_fisher_v(fisher_v)
        self.ita = ita
        self.damp = damp
        self._init_bias_corr()
        self._init_momentum()
        self._init_r(r)
        self.omega = omega
        if self.omega is None:
            self.omega = self.beta

    def _init_r(self, r):
        if r is None:
            self._fisher_r = tf.get_variable(
                self.w_name + "_r",
                initializer=tf.ones([self.shape[0], self.shape[1]]),
                trainable=False)
        else:
            self._fisher_r = r

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
    def v_b(self):
        pi = tf.sqrt((tf.trace(self.fisher_v) / self.shape[1] + 1e-8)
                     / (tf.trace(self.fisher_u) / self.shape[0] + 1e-8))
        coeff = self.lam / (self.N * self.ita)
        v_s, v_tmp = tf.self_adjoint_eig(self.fisher_v + pi * (coeff + self.damp) ** 0.5 * tf.eye(self.shape[1]))
        return v_tmp

    @property
    def u_b(self):
        pi = tf.sqrt((tf.trace(self.fisher_v) / self.shape[1] + 1e-8)
                     / (tf.trace(self.fisher_u) / self.shape[0] + 1e-8))
        coeff = self.lam / (self.N * self.ita)
        u_s, u_tmp = tf.self_adjoint_eig(self.fisher_u + 1. / pi * (coeff + self.damp) ** 0.5 * tf.eye(self.shape[0]))
        return u_tmp

    @property
    def fisher_u(self):
        return self._fisher_u

    @property
    def fisher_v(self):
        return self._fisher_v

    @property
    def fisher_r(self):
        return self._fisher_r

    @property
    def kfac_scale(self):
        pi = tf.sqrt((tf.trace(self.fisher_v) / self.shape[1] + 1e-8)
                     / (tf.trace(self.fisher_u) / self.shape[0] + 1e-8))
        coeff = self.lam / (self.N * self.ita)

        u_s, _ = tf.self_adjoint_eig(self.fisher_u +
                                     1. / pi * (coeff + self.damp) ** 0.5 *
                                     tf.eye(self.shape[0]))
        v_s, _ = tf.self_adjoint_eig(self.fisher_v + pi *
                                     (coeff + self.damp) ** 0.5 *
                                     tf.eye(self.shape[1]))
        # -> 'input_size' x 'input_size', 'output_size, output_size'
        u_s = tf.expand_dims(u_s, -1)
        v_s = tf.expand_dims(v_s, -1)
        return tf.matmul(u_s, v_s, transpose_b=True)

    def init_r_kfac(self):
        return self.fisher_r.assign(self.kfac_scale)

    @property
    def r(self):
        # Damp R here.
        coeff = self.lam / self.N
        var = coeff / (self.fisher_r + coeff / self.ita)
        return var

    def update(self, w, w_grad, a, s_grad):
        coeff = self.lam / (self.N * self.ita)

        u_b_tmp = self.u_b
        v_b_tmp = self.v_b
        r_tmp = self.fisher_r

        # Calculate gradient.
        gradient = tf.reduce_mean(w_grad - coeff * w, 0)
        ratio = 0.1
        new_momentum = self._momentum * (1. - ratio) + gradient * ratio
        new_bias_corr = self._bias_corr * (1. - ratio) + ratio

        # Project into Eigenbasis.
        # kron(ub, vb) * R^{-1} * kron(ub, vb)^T * (F * (DW - w))
        u_b_tmp_t = tf.transpose(u_b_tmp)
        mu_grad = tf.matmul(u_b_tmp_t,
                            tf.matmul(new_momentum / new_bias_corr, v_b_tmp))
        # Now divide by the rescaling matrix.
        damped_r = r_tmp + coeff + self.damp
        mu_grad = mu_grad / damped_r
        # One more kron product.
        v_b_tmp_t = tf.transpose(v_b_tmp)
        mu_grad = tf.matmul(u_b_tmp,
                            tf.matmul(mu_grad, v_b_tmp_t))
        # no_neg_r = tf.assert_non_negative(damped_r)
        # with tf.control_dependencies([no_neg_r]):
        infer1 = self._mu.assign(self._mu + self.alpha * mu_grad)

        # Update Momentum and Bias_corr; Same as Noisy MVG.
        infer4 = self._momentum.assign(new_momentum)
        infer5 = self._bias_corr.assign(new_bias_corr)
        return [infer1, infer4, infer5]

    def update_basis(self, w, w_grad, a, s_grad):
        # Update U; Same as Noisy MVG.
        a_t = tf.transpose(a, [0, 2, 1])
        fish_u = tf.reduce_mean(tf.matmul(a_t, a), [0]) / tf.to_float(tf.shape(a)[1])
        infer2 = self._fisher_u.assign((1. - self.beta) * self._fisher_u + self.beta * fish_u)

        # Update V; Same as Noisy MVG.
        s_grad_t = tf.transpose(s_grad, [0, 2, 1])
        fish_v = tf.reduce_mean(tf.matmul(s_grad_t, s_grad), [0]) / tf.to_float(tf.shape(s_grad)[1])
        infer3 = self._fisher_v.assign((1. - self.beta) * self._fisher_v + self.beta * fish_v)

        return [infer2, infer3]

    def update_scale(self, w, w_grad, a, s_grad):
        u_b_tmp = self.u_b
        v_b_tmp = self.v_b
        r_tmp = self.fisher_r

        # Update R; Different from Noisy MVG.
        # a: 'num_particles x batch_size x input_size'
        # s_grad: 'num_particles x batch_size x output_size'
        expanded_a = tf.reshape(a, [-1, tf.shape(a)[2]])
        expanded_s_grad = tf.reshape(s_grad, [-1, tf.shape(s_grad)[2]])
        # Make it to 'num_particles * batch_size x output_size'
        transformed_inputs = tf.matmul(expanded_a, u_b_tmp)
        transformed_outputs = tf.matmul(expanded_s_grad, v_b_tmp)
        # -> 'num_particles * batch_size x unit_size'
        reshaped_inputs = tf.reshape(transformed_inputs, tf.shape(a))
        reshaped_inputs_t = tf.transpose(reshaped_inputs, [0, 2, 1])
        reshaped_outputs = tf.reshape(transformed_outputs, tf.shape(s_grad))

        fish = tf.reduce_mean(tf.matmul(reshaped_inputs_t ** 2.,
                                        reshaped_outputs ** 2.), 0) / tf.to_float(tf.shape(a)[1])

        infer6 = self._fisher_r.assign((1 - self.omega) * self._fisher_r + self.omega * fish)

        return [infer6]