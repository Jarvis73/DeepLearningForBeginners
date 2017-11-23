#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process LIDC labels

@author: Jarvis ZHANG
@date: 2017/10/5
@framework: Tensorflow
@editor: VS Code
"""

import platform
import os


class PlatformError(Exception):
    def __init__(self, err='Need Windows or Linux'):
        super(PlatformError, self).__init__(err)


if 'Windows' in platform.system():
    SYS_ROOT_DIR = 'D:\\'
elif 'Linux' in platform.system():
    SYS_ROOT_DIR = '/home/jarvis'
else:
    raise PlatformError()


# Define dataset root dir
DATASET_ROOT_DIR = os.path.join(SYS_ROOT_DIR, "DataSet")

# Define Luna16 and SA root dir
LUNA16_ROOT_DIR = os.path.join(DATASET_ROOT_DIR, "Luna16")
SA_ROOT_DIR = os.path.join(DATASET_ROOT_DIR, "SA")

# Define Luna16 and LIDC dirs
LUNA16_RAW_SRC_DIR = os.path.join(LUNA16_ROOT_DIR, "data_set")
LIDC_XML_DIR = os.path.join(LUNA16_ROOT_DIR, "LIDC-XML-only")

# Define SA dirs
SA_SEG_DIR = os.path.join(SA_ROOT_DIR, "seg_anno")
SA_SEG_CUBE_DIR = os.path.join(SA_ROOT_DIR, "seg_cube")
SA_IMGS_DIR = os.path.join(SA_ROOT_DIR, "imgs")
SA_TEMP_DIR = os.path.join(SA_ROOT_DIR, "temp")

# Define logging path
MY_LOGGING_DIR = os.path.join(SYS_ROOT_DIR, "Logging")
SA_LOGGING_DIR = os.path.join(MY_LOGGING_DIR, "SA")
SA_LOG_TRAIN_DIR = os.path.join(SA_LOGGING_DIR, "train")
SA_LOG_TEST_DIR = os.path.join(SA_LOGGING_DIR, "test")
SA_LOG_TEST_IMG_DIR = os.path.join(SA_LOG_TEST_DIR, "image")
SA_LOG_TEST_DIR2 = os.path.join(SA_LOGGING_DIR, "test222")
SA_LOG_BEST_DIR = os.path.join(SA_LOGGING_DIR, "best_model")
SA_LOG_ALL_BEST_DIR = os.path.join(SA_LOGGING_DIR, "stored_best_model")
