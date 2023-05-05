import logging
import colorlog

def configure_logger(
        path: str = 'pyllamacpp.log',
        level: int = logging.INFO
    ):
    logger = logging.getLogger('pyllamacpp')
    logger.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a colorlog formatter with colorized log level names
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s%(reset)s - %(message)s',
        log_colors={
            'DEBUG': 'blue',
            'INFO': '',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(color_formatter)

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
