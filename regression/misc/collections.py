from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

_FLAGS = {}


def add_to_collection(key, value):
    _FLAGS[key] = value


def get_collection(key):
    return _FLAGS[key]
