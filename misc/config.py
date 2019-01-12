from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bunch import Bunch

import json
import os


class Bunch3(Bunch):
    def __repr__(self):
        keys = self.keys()
        args = ', '.join(['%s=%r' % (key, self[key]) for key in keys])
        return '%s(%s)' % (self.__class__.__name__, args)


def get_config_from_json(json_file):
    """ Get the config from a json file.
    :param json_file: String
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch3(config_dict)

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
