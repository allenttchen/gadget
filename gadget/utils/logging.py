import logging


def make_logger(
    name: str, 
    log_level: int = logging.DEBUG, 
    fmt: str = "[%(asctime)s|%(levelname)s] %(message)s", 
    datefmt: str = "%I:%M:%S", 
    remove_default_handlers: bool = True, 
    add_stream_handler: bool = True, 
    add_file_handler: bool = False, 
    filepath: str = None, 
    propagate=False, 
) -> logging.Logger:
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if remove_default_handlers:
        logger.handlers = []
        
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    
    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    if add_file_handler:
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = propagate
    
    return logger
