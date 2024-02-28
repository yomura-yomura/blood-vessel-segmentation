import logging as _logging

logger = _logging.getLogger(__name__)
logger.addHandler(_logging.NullHandler())

_handler = _logging.StreamHandler()
_handler.setFormatter(_logging.Formatter("[%(asctime)s|%(name)s|%(levelname)s] %(message)s"))
logger.addHandler(_handler)

logger.setLevel(_logging.NOTSET)
