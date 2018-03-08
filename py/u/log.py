import logging
import logging.handlers
import sys
import threading
from collections import defaultdict

from multiprocessing import Lock

from u.config import LOG_FOLDER, MAIN_THREAD, AVAILABLE_GPUS


class Log:
    logger = defaultdict(lambda k: Log.initiate_generic_logger(k))
    _FORCE_MAIN = len(AVAILABLE_GPUS) <= 1
    classlock = Lock()

    @staticmethod
    def l(force_main=False) -> logging:
        if Log._FORCE_MAIN or force_main:
            tid = MAIN_THREAD
        else:
            tid = threading.current_thread().name
        if tid not in Log.logger:
            Log.classlock.acquire()
            try:
                if tid not in Log.logger:
                    Log.logger[tid] = Log.initiate_generic_logger(tid)
            finally:
                Log.classlock.release()
        return Log.logger[tid]

    @staticmethod
    def initiate_generic_logger(thread_name):
        logger = logging.getLogger(thread_name)
        logger.setLevel(logging.INFO)
        if thread_name == MAIN_THREAD:
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.FileHandler(LOG_FOLDER + thread_name + ".log")
        formatter = logging.Formatter('%(name)s %(levelname)-8s %(relativeCreated)-6d	%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def info(*args, **kwargs):
        Log.l().info(*args, **kwargs)

    @staticmethod
    def var(**kwargs):
        Log.l().info("%s", kwargs.items())

    @staticmethod
    def vartime(arg):
        for e, tup in sorted(arg.items(), key=lambda t: t[1][0], reverse=True):
            Log.l().info("Time\t\t%s %.4f %.4f %.4f" % tuple([e] + list(tup)))

    @staticmethod
    def dvar(**kwargs):
        Log.l().debug("%s", kwargs)

    @staticmethod
    def debug(*args, **kwargs):
        Log.l().debug(*args, **kwargs)

    @staticmethod
    def critical(*args, **kwargs):
        Log.l().critical(*args, **kwargs)

    @staticmethod
    def exception(*args, **kwargs):
        Log.l().exception(*args, **kwargs)

    @staticmethod
    def info_main(*args, **kwargs):
        Log.l(True).info(*args, **kwargs)