# -*- coding:utf-8 -*-

import os
from datetime import date

PATH="log/"

def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
                os.makedirs(dir)

assure_path_exists(PATH)

PINGAN_LOGGING_CONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "pingan": {
            "format": "[%(asctime)s]- %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
        },
        "other": {
            "format": "[%(asctime)s] [%(levelname)s] [%(threadName)s:%(thread)d] [%(filename)s:%(lineno)d] [] - %(message)s"
        }

    },

    "handlers": {
        "pingan_console_handler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "pingan",
            "stream": "ext://sys.stdout"
        },
        "pingan_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "pingan",
            "filename": PATH +date.today().isoformat()+".log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
        "other_console_handler": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "other",
            "stream": "ext://sys.stdout"
        },
        "other_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "WARNING",
            "formatter": "other",
            "filename": PATH + date.today().isoformat() + ".log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },

    },
    'loggers': {
        # "sc": {
        #     "level": "INFO",
        #     "handlers": ["sc_console_handler", "sc_file_handler"]
        # },
        "": {
            "level": "INFO",
            "handlers": ["pingan_console_handler", "pingan_file_handler"]
        }
    }

}
