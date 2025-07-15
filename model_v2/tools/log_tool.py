# -*- coding: utf-8 -*-
# @File    : log_tool.py
# @Author  : juntang
# @Time    : 2022/11/16 14:11

import logging
import os
import time


def log_tool_init(model_start_time, crime_id, path, note="",
                  level=logging.INFO,
                  console_level=logging.INFO,
                  no_console=False):

    # clear handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []

    # define log dir path & log file path
    if note == "":
        log_path = os.path.join(path, "log_crime-{}_{}".format(crime_id, model_start_time))
    else:
        log_path = os.path.join(path, "log_crime-{}_{}_{}".format(crime_id, note, model_start_time))

    # make log handler
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(log_path)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        log_console = logging.StreamHandler()
        log_console.setLevel(console_level)
        log_console.setFormatter(formatter)
        logging.root.addHandler(log_console)
