from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from classification.ops.optimizer import NGOptimizer
from classification.ops import layer_collection as lc
from classification.controller.weight_controller import WeightController
from misc.registry import get_model
from core.base_model import BaseModel
from classification.network.vgg import *


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
            print("[!] Optimizer: EKFAC")
            self.layer_collection = lc.LayerCollection(mode="ekfac")
        elif config.optimizer == "kfac":
            print("[!] Optimizer: KFAC")
            self.layer_collection = lc.LayerCollection(mode="kfac")
        else:
            print("[!] Optimizer: {}".format(config.optimizer))
            self.layer_collection = None

        self.input_dim = input_dim
        self.n_data = n_data

        # Define Operations.
        self.re_init_kfac_scale_op = None
        self.cov_update_op = None
        self.inv_update_op = None
        self.eigen_basis_update_op = None
        self.scale_update_op = None
        self.var_update_op = None
        self.var_scale_update_op = None
        self.train_op = None

        # Initialize attributes.
        self.inputs = None
        self.targets = None
        self.is_training = None
        self.n_particles = None
        self.sampler = None
        self.acc = None
        self.loss = None
        self.total_loss = None
        self.optim = None
        self.saver = None

        # Build the model.
        self.build_model()
        self.init_optim()
        self.init_saver()

    @property
    def trainable_variables(self):
        # Don't train the params of BN.
        vars_ = []
        for var in tf.trainable_variables():
            # Get either weight or bias.
            if "w" in var.name or "bias" in var.name:
                vars_.append(var)
        return vars_

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [None] + self.input_dim)
        self.targets = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)
        self.n_particles = tf.placeholder(tf.int32)

        inputs = self.inputs
        net = get_model(self.config.model_name)

        # Initialize a sampler.
        self.sampler = WeightController(self.n_data, self.config, self.n_particles)
        logits, l2_loss = net(inputs, self.sampler, self.is_training,
                              self.config.batch_norm, self.layer_collection,
                              self.n_particles)

        # Ensemble from n_particles.
        logits_ = tf.reduce_mean(
            tf.reshape(logits, [self.n_particles, -1, tf.shape(logits)[-1]]), 0)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(
            self.targets, tf.argmax(logits_, axis=1)), dtype=tf.float32))

        targets_ = tf.tile(self.targets, [self.n_particles])
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets_, logits=logits))

        # coeff = kl / (N * eta)
        coeff = self.config.kl / (self.n_data * self.config.eta)
        self.total_loss = self.loss + coeff * l2_loss

    def init_optim(self):
        if self.config.optimizer == "ekfac":
            self.optim = NGOptimizer(var_list=self.trainable_variables,
                                     learning_rate=tf.train.exponential_decay(self.config.learning_rate,
                                                                              self.global_step_tensor,
                                                                              self.config.decay_every_itr, 0.1,
                                                                              staircase=True),
                                     cov_ema_decay=self.config.cov_ema_decay,
                                     scale_ema_decay=self.config.scale_ema_decay,
                                     damping=self.config.damping,
                                     layer_collection=self.layer_collection,
                                     norm_constraint=tf.train.exponential_decay(self.config.kl_clip,
                                                                                self.global_step_tensor,
                                                                                390, 0.95, staircase=True),
                                     momentum=self.config.momentum,
                                     opt_type=self.config.optimizer)

            self.cov_update_op = self.optim.cov_update_op
            self.re_init_kfac_scale_op = self.optim.re_init_kfac_scale_op
            self.inv_update_op = None
            self.eigen_basis_update_op = self.optim.eigen_basis_update_op

            with tf.control_dependencies([self.eigen_basis_update_op]):
                self.var_update_op = self.sampler.update_weights(self.layer_collection.get_blocks())

            self.scale_update_op = self.optim.scale_update_op

            with tf.control_dependencies([self.scale_update_op]):
                self.var_scale_update_op = self.sampler.update_scales(self.layer_collection.get_blocks())

        if self.config.optimizer == "kfac":
            self.optim = NGOptimizer(var_list=self.trainable_variables,
                                     learning_rate=tf.train.exponential_decay(self.config.learning_rate,
                                                                              self.global_step_tensor,
                                                                              self.config.decay_every_itr, 0.1,
                                                                              staircase=True),
                                     cov_ema_decay=self.config.cov_ema_decay,
                                     damping=self.config.damping,
                                     layer_collection=self.layer_collection,
                                     norm_constraint=tf.train.exponential_decay(self.config.kl_clip,
                                                                                self.global_step_tensor,
                                                                                390, 0.95, staircase=True),
                                     momentum=self.config.momentum,
                                     opt_type=self.config.optimizer)

            self.cov_update_op = self.optim.cov_update_op
            self.eigen_basis_update_op = None
            self.scale_update_op = None
            self.inv_update_op = self.optim.inv_update_op

            with tf.control_dependencies([self.inv_update_op]):
                self.var_update_op = self.sampler.update_weights(self.layer_collection.get_blocks())

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optim.minimize(self.total_loss, global_step=self.global_step_tensor)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
