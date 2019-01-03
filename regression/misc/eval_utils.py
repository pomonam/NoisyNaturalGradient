from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import zhusuan as zs


def rmse(y_pred, y, std_y_train=1.):
    """ Root Mean Squared Error (RMSE)
    :param y_pred: tensor of shape [batch_size, ...]
    :param y: tensor of shape [batch_size, ...]
    :param std_y_train: float
    :return : tensor of shape []. RMSE.
    """
    return tf.sqrt(tf.reduce_mean((y_pred - y) ** 2)) * std_y_train


def log_likelihood(log_py_xw, std_y_train):
    """ Log Likelihood.
    :param log_py_xw: [n_particles, batch_size] or [batch_size]
    :param std_y_train: float
    :return : tensor of shape []. RMSE.
    """
    rank = len(log_py_xw.get_shape())
    if rank == 1:
        log_py_xw = tf.expand_dims(log_py_xw, [0])
    ll = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0)) - tf.log(std_y_train)
    return ll
