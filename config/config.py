# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: config.py
@ time: $17-7-18 下午5:58
"""

import os
import sys


# golbal params
THUNLP_MODEL_PATH = "/home/showlove/cc/code/THULAC-Python/models"
THUNLP_USER_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/user_dic.txt"
TEXT_SAMPLE_DIR = "/home/showlove/PycharmProjects/data_test/nlp/doc/corpus_6_4000"
STOP_WORD_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/stop_word_dic.txt"
DICTIONARY_PATH = "/home/showlove/PycharmProjects/data_test/tmp/sample.dict"
TFIDF_PATH = "/home/showlove/PycharmProjects/data_test/tmp/tfidf_corpus"
LSI_PATH = "/home/showlove/PycharmProjects/data_test/tmp/LSI_corpus"
LSI_MODEL = "/home/showlove/PycharmProjects/data_test/tmp/lsi_model.pkl"
PREDICTOR_MODEL = "/home/showlove/PycharmProjects/data_test/tmp/predictor.pkl"
CLASS_NUM = 6
SAMPLE_NUM = 4000