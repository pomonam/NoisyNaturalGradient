from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from core.base_model import BaseModel
from misc.registry import get_model
from regression.controller.bayesian_learning import BayesianLearning
from regression.misc.layers import *
from regression.controller.sample import NormalOutSample
from regression.network.ffn import *


class Model(BaseModel):
    def __init__(self, config, input_dim, n_data):
        """ Initialize a class Model.
        :param config: Configuration Bundle.
        :param input_dim: int
        :param n_data: int
        """
        super().__init__(config)
        # Set the approximation type specifically.
        if config.optimizer == "ekfac":
            self.layer_type = "emvg"
        elif config.optimizer == "kfac":
            print("[!] Optimizer: KFAC")
            self.layer_type = "mvg"
        else:
            print("[!] Optimizer: {}".format(config.optimizer))
            self.layer_type = None
        self.input_dim = input_dim
        self.n_data = n_data

        # Define Operations.
        self.init_ops = None
        self.basis_update_op = None
        self.scale_update_op = None
        self.train_op = None

        # Initialize attributes.
        self.inputs = None
        self.targets = None
        self.is_training = None
        self.n_particles = None
        self.saver = None
        self.rmse = None
        self.ll = None
        self.lower_bound = None
        self.log_py_xw = None
        self.kl = None
        self.y_pred = None
        self.y = None
        self.loss_prec = None
        self.alpha = None
        self.beta = None
        self.omega = None

        # Build the model.
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [None] + self.input_dim)
        self.targets = tf.placeholder(tf.float32, [None])
        self.n_particles = tf.placeholder(tf.int32)
        self.alpha = tf.placeholder(tf.float32, shape=[], name='alpha')
        self.beta = tf.placeholder(tf.float32, shape=[], name='beta')
        self.omega = tf.placeholder(tf.float32, shape=[], name='omega')

        inputs = self.inputs
        net = get_model(self.config.model_name)

        layers, init_ops, num_hidden = net(self.layer_type,
                                           int(inputs.shape[-1]),
                                           self.n_data,
                                           self.config.kl,
                                           self.config.eta,
                                           self.alpha,
                                           self.beta,
                                           self.config.damping,
                                           self.omega)

        if self.layer_type == "emvg":
            learn = BayesianLearning([inputs.shape[-1], num_hidden, 1], [EMVGLayer] * 2, [{}] * 2,
                                     {}, tf.nn.relu, NormalOutSample,
                                     self.inputs, self.targets, self.n_particles,
                                     std_y_train=self.config.std_train)

        elif self.layer_type == "mvg":
            learn = BayesianLearning([inputs.shape[-1], 50, 1], [MVGLayer] * 2, [{}] * 2,
                                     {}, tf.nn.relu, NormalOutSample,
                                     self.inputs, self.targets, self.n_particles,
                                     std_y_train=self.config.std_train)

        else:
            raise NotImplementedError()

        log_py_xw = learn.log_py_xw
        kl = learn.kl
        lower_bound = tf.reduce_mean(log_py_xw - self.config.kl * kl / self.n_data)

        log_alpha, log_beta = tf.trainable_variables()[-2:]
        alpha_ = tf.exp(log_alpha)
        beta_ = tf.exp(log_beta)
        y_obs = tf.tile(tf.expand_dims(self.targets, 0), [self.n_particles, 1])
        h_pred = tf.squeeze(learn.h_pred, 2)
        loss_prec = 0.5 * (tf.stop_gradient(tf.reduce_mean((y_obs - h_pred) ** 2)) *
                           alpha_ / beta_ - (tf.digamma(alpha_) - tf.log(beta_ + 1e-10)))

        lower_bound = lower_bound - tf.reduce_mean(loss_prec)
        self.lower_bound = lower_bound

        vars = []
        for var in tf.trainable_variables():
            if 'log' in var.name:
                vars.append(var)

        infer = []
        qws = [learn.qws['w' + str(i)].tensor for i in range(len(learn.qws))]
        w_grads = tf.gradients(tf.reduce_sum(tf.reduce_mean(log_py_xw, 1), 0), qws)

        activations = [get_collection("a0"), get_collection("a1")]
        activations = [tf.concat(
            [activation,
             tf.ones(tf.concat([tf.shape(activation)[:-1], [1]], axis=0))], axis=-1) for activation in activations]

        s = [get_collection("s0"), get_collection("s1")]

        if self.config.true_fisher:
            sampled_log_prob = learn.sampled_log_prob
            s_grads = tf.gradients(tf.reduce_sum(sampled_log_prob), s)
        else:
            s_grads = tf.gradients(tf.reduce_sum(log_py_xw), s)

        scale_update_ops = []
        basis_update_ops = []
        for l, w, w_grad, a, s_grad in zip(layers, qws, w_grads, activations, s_grads):
            infer = infer + l.update(w, w_grad, a, s_grad)
            if self.layer_type == "emvg":
                scale_update_ops = scale_update_ops + l.update_scale(w, w_grad, a, s_grad)
                basis_update_ops = basis_update_ops + l.update_basis(w, w_grad, a, s_grad)

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        grads = optimizer.compute_gradients(-lower_bound, vars)
        infer = infer + [optimizer.apply_gradients(grads)]

        self.train_op = infer
        self.init_ops = tf.group(init_ops) if init_ops != [] else None
        self.basis_update_op = tf.group(*basis_update_ops) if basis_update_ops != [] else None
        self.scale_update_op = tf.group(*scale_update_ops) if scale_update_ops != [] else None

        self.rmse = learn.rmse
        self.ll = learn.log_likelihood
        self.lower_bound = lower_bound
        self.log_py_xw = tf.reduce_mean(learn.log_py_xw)
        self.kl = learn.kl
        self.y_pred = tf.reduce_mean(learn.h_pred)
        self.y = tf.reduce_mean(self.targets)
        self.loss_prec = loss_prec

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
