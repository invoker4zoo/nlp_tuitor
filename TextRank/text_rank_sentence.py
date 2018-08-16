# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: text_rank_sentence.py
@ time: $18-8-15 下午2:26
"""
import numpy as np
from tool.logger import logger
from tool.punct import punct
import thulac

# GOLBAL PARAMS
THUNLP_MODEL_PATH = "/home/showlove/cc/code/THULAC-Python/models"
THUNLP_USER_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/user_dic.txt"
STOP_WORD_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/stop_word_dic.txt"
VECTOR_MODEL = "/home/showlove/cc/nlp/vector/sgns.renmin.bigram-char"

class TextRank(object):

    def __init__(self):
        pass