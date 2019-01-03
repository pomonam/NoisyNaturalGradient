from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from misc.utils import Summarizer

import tensorflow as tf


class BaseTrain:
    """ An abstract class for trainers. Children include trainers for classification and regression."""
    def __init__(self, sess, model, config, logger):
        self.model = model
        self.logger = logger
        self.summarizer = Summarizer(sess, config)
        self.config = config
        self.sess = sess
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.config.epoch):
            self.logger.info('epoch: {}'.format(int(cur_epoch)))
            self.train_epoch()
            self.test_epoch()

    def train_epoch(self):
        """ Implement the logic of epoch:
        - Loop over the number of iterations in the config and call the train step.
        - Add any summaries you want using the summary.
        """
        raise NotImplementedError()

    def test_epoch(self):
        """ Implement the logic of the train step.
        - Run the TensorFlow session.
        - Return any metrics you need to summarize.
        """
        raise NotImplementedError()
