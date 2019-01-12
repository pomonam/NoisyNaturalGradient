from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.base_train import BaseTrain
from tqdm import tqdm

import numpy as np


class Trainer(BaseTrain):
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch()
            self.test_epoch()

            if (cur_epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.model.save(self.sess)

    def train_epoch(self):
        loss_list = []
        acc_list = []
        for itr, (x, y) in enumerate(tqdm(self.train_loader)):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.n_particles: self.config.train_particles
            }
            feed_dict.update({self.model.is_training: True})
            self.sess.run([self.model.train_op], feed_dict=feed_dict)

            feed_dict.update({self.model.is_training: False})  # Note: that's important
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

            cur_iter = self.model.global_step_tensor.eval(self.sess)

            if cur_iter % self.config.get('TCov', 10) == 0 and \
                    self.model.cov_update_op is not None:
                self.sess.run([self.model.cov_update_op], feed_dict=feed_dict)

                if self.config.optimizer == "diag":
                    self.sess.run([self.model.var_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.get('TInv', 200) == 0 and \
                    self.model.inv_update_op is not None:
                self.sess.run([self.model.inv_update_op, self.model.var_update_op], feed_dict=feed_dict)

            if cur_iter % self.config.get('TEigen', 200) == 0 and \
                    self.model.eigen_basis_update_op is not None:
                self.sess.run([self.model.eigen_basis_update_op, self.model.var_update_op], feed_dict=feed_dict)

                if self.config.kfac_init_after_basis:
                    self.sess.run(self.model.re_init_kfac_scale_op)

            if cur_iter % self.config.get('TScale', 10) == 0 and \
                    self.model.scale_update_op is not None:
                self.sess.run([self.model.scale_update_op, self.model.var_scale_update_op], feed_dict=feed_dict)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("train | loss: %5.6f | accuracy: %5.6f" % (float(avg_loss), float(avg_acc)))

        # Summarize
        summaries_dict = dict()
        summaries_dict['train_loss'] = avg_loss
        summaries_dict['train_acc'] = avg_acc

        # Summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

    def test_epoch(self):
        loss_list = []
        acc_list = []
        for (x, y) in self.test_loader:
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.is_training: False,
                self.model.n_particles: self.config.test_particles
            }
            loss, acc = self.sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            loss_list.append(loss)
            acc_list.append(acc)

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info("test | loss: %5.6f | accuracy: %5.6f\n" % (float(avg_loss), float(avg_acc)))

        # Summarize
        summaries_dict = dict()
        summaries_dict['test_loss'] = avg_loss
        summaries_dict['test_acc'] = avg_acc

        # Summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)
