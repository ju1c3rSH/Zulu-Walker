import logging

_logger = logging.getLogger("framework")


def fw_log(msg: str = "", *args) -> None:
    if args:
        msg = " ".join([str(msg)] + [str(a) for a in args])
    _logger.info(msg)
