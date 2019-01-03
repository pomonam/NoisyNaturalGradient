from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


class NGOptimizer(object):
    def __init__(self, shape, N, lam, alpha, beta, w_name):
        self.shape = shape
        self.N = N
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.w_name = w_name

    def update(self, w, w_grad, a, s_grad):
        raise NotImplementedError()
