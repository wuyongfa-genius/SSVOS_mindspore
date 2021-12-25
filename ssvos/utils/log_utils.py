"""Log some info into console or file."""
import logging
from mindspore.log import _get_logger


def set_logger_level_to(logger=_get_logger(), level=logging.INFO):
    logger.setLevel(level)


def master_only_info(msg=None, rank=0, logger=_get_logger(), *args, **kwargs):
    if rank==0:
        logger.info(msg, *args, **kwargs)

