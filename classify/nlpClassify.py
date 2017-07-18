# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: nlpClassify.py
@ time: $17-7-18 下午7:16
"""

import os
import sys
import thulac
from nlp.punct import punct
from gensim import corpora, models
import time
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm


