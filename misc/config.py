from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bunch import Bunch

import json
import os


def get_config_from_json(json_file):
    """ Get the config from a json file.
    :param json_file: String
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    # may cause error in Python 3 >.
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    """ Process the configuration file.
    :param json_file: String
    :return: config(dictionary)
    """
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join("./experiments", config.dataset, config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("./experiments", config.dataset, config.exp_name, "checkpoint/")
    return config
