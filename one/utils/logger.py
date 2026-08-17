import logging
import os


def setup_logger(name='one', level='INFO', log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
