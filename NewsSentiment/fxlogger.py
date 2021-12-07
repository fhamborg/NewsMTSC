import logging

logger = None


def get_logger():
    global logger

    if not logger:
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(module)s(%(lineno)d):%(funcName)s|: %(message)s',
                                      datefmt='%H:%M:%S')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger = logging.getLogger("NewsSentiment")
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.ERROR)

        if len(logger.handlers) == 0:
            logger.addHandler(handler)

    return logger
