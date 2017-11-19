# Copyright 2017 Bruno Ribeiro, Mayank Kakodkar, Pedro Savarese
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import logging.handlers

import sys


class Log:
    logger = None

    @staticmethod
    def l() -> logging:
        if Log.logger is None:
            Log.logger = Log.initiate_generic_logger(logger_name="RBM")
        return Log.logger

    @staticmethod
    def initiate_generic_logger(logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(name)s %(levelname)-8s %(relativeCreated)-6d	%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def info(*args, **kwargs):
        Log.l().info(*args, **kwargs)

    @staticmethod
    def var(**kwargs):
        Log.l().info("%s", list(sorted(kwargs.items())))

    @staticmethod
    def dvar(**kwargs):
        Log.l().debug("%s", kwargs)

    @staticmethod
    def debug(*args, **kwargs):
        Log.l().debug(*args, **kwargs)
