from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from misc.utils import get_logger, get_args, makedirs
from misc.config import process_config
from regression.misc.data_loader import generate_data_loader
from regression.train import Trainer as RegressionTrainer
from regression.model import Model as RegressionModel

import tensorflow as tf
import numpy as np
import os

_REGRESSION_INPUT_DIM = {
    "concrete": [8],
    "energy": [8],
    "housing": [13],
    "kin8nm": [8],
    "naval": [17],
    "power_plant": [4],
    "protein": [9],
    "wine": [11],
    "yacht_hydrodynamics": [6],
    "year_prediction": [90]
}


def main():
    tf.set_random_seed(1231)
    np.random.seed(1231)

    config = None
    try:
        args = get_args()
        config = process_config(args.config)

        if config is None:
            raise Exception()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path3 = os.path.join(path, 'regression/model.py')
    path4 = os.path.join(path, 'regression/train.py')
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=os.path.abspath(__file__), package_files=[path3, path4])

    logger.info(config)

    # Define computational graph
    rmse_results, ll_results = [], []
    n_runs = 10

    for i in range(1, n_runs + 1):
        sess = tf.Session()

        # Perform data splitting again with the provided seed.
        train_loader, test_loader, std_train = generate_data_loader(config, seed=i)
        config.std_train = std_train

        model_ = RegressionModel(config,
                                 _REGRESSION_INPUT_DIM[config.dataset],
                                 len(train_loader.dataset))
        trainer = RegressionTrainer(sess, model_, train_loader, test_loader, config, logger)

        trainer.train()

        rmse, ll = trainer.get_result()

        rmse_results.append(float(rmse))
        ll_results.append(float(ll))

        tf.reset_default_graph()

    for i, (rmse_result, ll_result) in enumerate(zip(rmse_results,
                                                     ll_results)):
        logger.info("\n## RUN {}".format(i))
        logger.info('# Test rmse = {}'.format(rmse_result))
        logger.info('# Test log likelihood = {}'.format(ll_result))

    logger.info("Results (mean/std. errors):")
    logger.info("Test rmse = {}/{}".format(
        np.mean(rmse_results), np.std(rmse_results) / n_runs ** 0.5))
    logger.info("Test log likelihood = {}/{}".format(
        np.mean(ll_results), np.std(ll_results) / n_runs ** 0.5))


if __name__ == "__main__":
    main()
