import logging
import time


def log_exec_time(logger=logging, log_level=logging.INFO, action_name=None):
    def log_exec_time_decorator(fn):
        def wrapper(*args, **kwargs):
            started = time.time()
            res = fn(*args, **kwargs)
            ended = time.time()
            action = action_name if not action_name == None else fn.__name__
            logger.log(log_level, "Action {} took {} seconds to execute.".format(action, started - ended))

            return res

        return wrapper

    return log_exec_time_decorator