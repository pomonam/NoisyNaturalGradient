from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from core.base_train import BaseTrain

import numpy as np


class Trainer(BaseTrain):
    def __init__(self, sess, model, train_loader, test_loader, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.alpha = self.model.config.alpha
        self.beta = self.model.config.beta
        self.omega = self.model.config.omega

    def train(self):
        if self.model.init_ops is not None:
            self.sess.run(self.model.init_ops)

        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch()

            if cur_epoch % self.config.get("epoch_rate", 10) == 0:
                self.test_epoch()

            if (cur_epoch + 1) % self.config.get('save_interval', 1000) == 0:
                self.model.save(self.sess)

            if (cur_epoch + 1) % self.config.get("lr_decay_interval", 500) == 0:
                decay_ratio = self.config.get("lr_decay_ratio", 0.1)
                self.alpha = decay_ratio * self.alpha
                self.beta = decay_ratio * self.beta
                self.omega = decay_ratio * self.omega

    def train_epoch(self):
        lb_lst = []
        log_py_xw_list = []
        kl_list = []
        y_pred_list = []
        y_list = []
        loss_prec_list = []

        for itr, (x, y) in enumerate(self.train_loader):
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.n_particles: self.config.train_particles,
                self.model.alpha: self.alpha,
                self.model.beta: self.beta,
                self.model.omega: self.omega
            }
            self.sess.run([self.model.train_op], feed_dict=feed_dict)

            cur_iter = self.model.global_step_tensor.eval(self.sess)

            if self.config.amor_eigen:
                if cur_iter % self.config.get("Teigen", 5) == 0:
                    self.sess.run([self.model.basis_update_op], feed_dict=feed_dict)
                    self.sess.run([self.model.init_ops])
            elif self.config.optimizer == "ekfac":
                self.sess.run([self.model.basis_update_op], feed_dict=feed_dict)

            if self.config.re_init:
                if cur_iter % self.config.get('re_init_iters', 50) == 0:
                    self.sess.run([self.model.init_ops])

            if self.model.scale_update_op is not None:
                self.sess.run([self.model.scale_update_op], feed_dict=feed_dict)

            lb, log_py_xw, kl, y_pred, y, loss_prec = self.sess.run([self.model.lower_bound,
                                                                     self.model.log_py_xw,
                                                                     self.model.kl,
                                                                     self.model.y_pred,
                                                                     self.model.y,
                                                                     self.model.loss_prec], feed_dict=feed_dict)
            lb_lst.append(lb)
            log_py_xw_list.append(log_py_xw)
            kl_list.append(kl)
            y_pred_list.append(y_pred)
            y_list.append(y)
            loss_prec_list.append(loss_prec)

        average_lb = np.mean(lb_lst)
        average_log_py_xw = np.mean(log_py_xw_list)
        average_kl = np.mean(kl_list)
        average_y_pred = np.mean(y_pred_list)
        average_y = np.mean(y_list)
        average_loss_prec = np.mean(loss_prec_list)

        self.logger.info("train | Lower Bound: %5.6f | log_py_wx: %5.6f | "
                         "KL: %5.6f | y_pred: %5.6f | y: %5.6f | "
                         "loss prec: %5.6f" % (float(average_lb),
                                               float(average_log_py_xw),
                                               float(average_kl),
                                               float(average_y_pred),
                                               float(average_y),
                                               float(average_loss_prec)))

        # Summarize
        summaries_dict = dict()
        summaries_dict['train_lb'] = average_lb
        summaries_dict['train_log_py_xw'] = average_log_py_xw
        summaries_dict['train_kl'] = average_kl
        summaries_dict['train_y_pred'] = average_y_pred
        summaries_dict['train_y'] = average_y
        summaries_dict['train_loss_prec'] = average_loss_prec

        # Summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

        # Shuffle the dataset.
        self.train_loader.dataset.permute(0)

    def test_epoch(self):
        lb_list = []
        rmse_list = []
        ll_list = []
        for (x, y) in self.test_loader:
            feed_dict = {
                self.model.inputs: x,
                self.model.targets: y,
                self.model.n_particles: self.config.train_particles,
                self.model.alpha: self.alpha,
                self.model.beta: self.beta,
                self.model.omega: self.omega
            }
            lb, rmse, ll = self.sess.run([self.model.lower_bound, self.model.rmse, self.model.ll], feed_dict=feed_dict)

            lb_list.append(lb)
            rmse_list.append(rmse)
            ll_list.append(ll)

        average_lb = np.mean(lb_list)
        average_rmse = np.mean(rmse_list)
        average_ll = np.mean(ll_list)
        self.logger.info("test | Lower Bound: %5.6f | RMSE: %5.6f | "
                         "Log Likelihood : %5.6f" % (float(average_lb), float(average_rmse), float(average_ll)))

        # Summarize
        summaries_dict = dict()
        summaries_dict['test_lower_bound'] = average_lb
        summaries_dict['test_rmse'] = average_rmse
        summaries_dict['test_log_likelihood'] = average_ll

        # Summarize
        cur_iter = self.model.global_step_tensor.eval(self.sess)
        self.summarizer.summarize(cur_iter, summaries_dict=summaries_dict)

        return average_rmse, average_ll

    def get_result(self):
        return self.test_epoch()
