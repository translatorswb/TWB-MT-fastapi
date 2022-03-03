import logging as stdlib_logging


def get_default_logger():
    return stdlib_logging.getLogger('console_logger')


logging = get_default_logger()


