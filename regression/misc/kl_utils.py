from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

distributions = tf.distributions


def mf_mf(param1, param2):
    mean1, logstd1 = param1[0], param1[1]
    mean2, logstd2 = param2[0], param2[1]
    return tf.reduce_sum(distributions.kl_divergence(distributions.Normal(mean1, tf.exp(logstd1)),
                                                     distributions.Normal(mean2, tf.exp(logstd2))))


def mvg_mf(param1, param2):
    mean1, u1, v1 = param1[0], param1[1], param1[2]
    mean2, logstd2 = param2[0], param2[1]

    n, p = tf.to_float(tf.shape(mean1)[0]), tf.to_float(tf.shape(mean1)[1])
    eu, _ = tf.self_adjoint_eig(u1)
    ev, _ = tf.self_adjoint_eig(v1)
    logdet_u = tf.reduce_sum(tf.log(eu), [-1])
    logdet_v = tf.reduce_sum(tf.log(ev), [-1])

    diag_u = tf.matrix_diag_part(u1)
    diag_v = tf.matrix_diag_part(v1)
    diag_uv = tf.matmul(tf.expand_dims(diag_u, -1),
                        tf.expand_dims(diag_v, -2))

    kl = 2. * tf.reduce_sum(logstd2, [-1, -2]) - n * logdet_v - p * logdet_u
    kl = kl - n * p + tf.reduce_sum(tf.multiply(diag_uv, tf.exp(-2. * logstd2)), [-1, -2])
    kl = kl + tf.reduce_sum(tf.square(mean2 - mean1) * tf.exp(-2. * logstd2), [-1, -2])
    return kl * 0.5


def emvg_mf(param1, param2):
    mean1, u1, v1, r = param1[0], param1[1], param1[2], param1[3]
    mean2, logstd2 = param2[0], param2[1]

    n, p = tf.to_float(tf.shape(mean1)[0]), tf.to_float(tf.shape(mean1)[1])

    logdet_r = tf.reduce_sum(tf.log(r), [-1, -2])

    kl = 2. * tf.reduce_sum(logstd2, [-1, -2]) - logdet_r
    kl = kl - n * p + tf.reduce_sum(tf.multiply(r, tf.exp(-2. * logstd2)), [-1, -2])
    kl = kl + tf.reduce_sum(tf.square(mean2 - mean1) * tf.exp(-2. * logstd2), [-1, -2])

    return kl * 0.5


def fg_mf_gamma(param1, param2):
    mean1, var = param1[0], param1[1]
    mean2, alpha, beta = param2[0], param2[1], param2[2]
    ab_ratio = alpha / beta

    n, p = tf.to_float(tf.shape(mean1)[0]), tf.to_float(tf.shape(mean1)[1])
    e, _ = tf.self_adjoint_eig(var)
    logdet_var = tf.reduce_sum(tf.log(e), [-1])

    kl = - n * p * (tf.digamma(alpha) - tf.log(beta+1e-10)) - logdet_var
    kl = kl - n * p + ab_ratio * tf.trace(var)
    kl = kl + ab_ratio * tf.reduce_sum(tf.square(mean2 - mean1), [-1, -2])
    return kl * 0.5


def mf_mf_gamma(param1, param2):
    mean1, logstd1 = param1[0], param1[1]
    mean2, alpha, beta = param2[0], param2[1], param2[2]
    ab_ratio = alpha / beta
    n, p = tf.to_float(tf.shape(mean1)[0]), tf.to_float(tf.shape(mean1)[1])

    kl = - n * p * (tf.digamma(alpha) - tf.log(beta)) - 2 * tf.reduce_sum(logstd1, [-1, -2])
    kl = kl - n * p + ab_ratio * tf.reduce_sum(tf.exp(logstd1 * 2), [-1, -2])
    kl = kl + ab_ratio * tf.reduce_sum(tf.square(mean2 - mean1), [-1, -2])
    return kl * 0.5


def mvg_mf_gamma(param1, param2):
    mean1, u1, v1 = param1[0], param1[1], param1[2]
    mean2, alpha, beta = param2[0], param2[1], param2[2]
    ab_ratio = alpha / beta

    n, p = tf.to_float(tf.shape(mean1)[0]), tf.to_float(tf.shape(mean1)[1])
    eu, _ = tf.self_adjoint_eig(u1)
    ev, _ = tf.self_adjoint_eig(v1)
    logdet_u = tf.reduce_sum(tf.log(eu), [-1])
    logdet_v = tf.reduce_sum(tf.log(ev), [-1])

    diag_u = tf.matrix_diag_part(u1)
    diag_v = tf.matrix_diag_part(v1)
    diag_uv = tf.matmul(tf.expand_dims(diag_u, -1),
                        tf.expand_dims(diag_v, -2))

    kl = - n * p * (tf.digamma(alpha) - tf.log(beta+1e-10)) - n * logdet_v - p * logdet_u
    kl = kl - n * p + ab_ratio * tf.reduce_sum(diag_uv, [-1, -2])
    kl = kl + ab_ratio * tf.reduce_sum(tf.square(mean2 - mean1), [-1, -2])
    return kl * 0.5


def gamma_gamma(param1, param2):
    alpha1, beta1 = param1[0], param1[1]
    alpha2, beta2 = param2[0], param2[1]
    return tf.reduce_sum(distributions.kl_divergence(
        distributions.Gamma(alpha1, beta1),
        distributions.Gamma(alpha2, beta2)))
