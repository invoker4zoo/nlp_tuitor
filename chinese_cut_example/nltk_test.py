# coding=utf-8
"""
nltk需要载入数据文件和java模型文件
path_to_dict
path_to_sihan
path_to_jar
path_to_model
java_class='edu.stanford.nlp.ie.crf.CRFClassifier'
"""
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import sys
sys.path.append('/usr/local/stanford-segmenter-2017-06-09')

path_to_jar = '/usr/local/stanford-segmenter-2017-06-09/stanford-segmenter-3.8.0.jar'
path_to_sihan_corpora_dict = "/usr/local/stanford-segmenter-2017-06-09/data"
path_to_model = "/usr/local/stanford-segmenter-2017-06-09/data/pku.gz"
path_to_dict = "/usr/local/stanford-segmenter-2017-06-09/data/dict-chris6.ser.gz"
segmenter = StanfordSegmenter(path_to_dict=path_to_dict, path_to_sihan_corpora_dict=path_to_sihan_corpora_dict,
                              path_to_jar=path_to_jar, path_to_model=path_to_model, java_class='edu.stanford.nlp.ie.crf.CRFClassifier')
sentence = u'这是斯坦福中文分词器测试'
list = segmenter.segment(sentence)
print list

