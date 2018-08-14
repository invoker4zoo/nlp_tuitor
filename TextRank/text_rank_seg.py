# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: text_rank_seg.py
@ time: $18-8-14 上午11:48
"""
import numpy as np
from tool.logger import logger
from tool.punct import punct
import thulac

# GOLBAL PARAMS
THUNLP_MODEL_PATH = "/home/showlove/cc/code/THULAC-Python/models"
THUNLP_USER_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/user_dic.txt"
STOP_WORD_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/stop_word_dic.txt"


class TextRank(object):
    """

    """
    def __init__(self, doc, window_size, alpha, step, doc_seg=False):
        """

        :param doc:
        :param window_size:
        :param alpha:
        :param step:
        :param doc_seg: 传入文档是否为分词后的文档
        """
        self.doc = doc
        self.window_size = window_size
        self.alpha = alpha
        self.step = step
        self.net_edge = {}
        # 分词/清洗
        if not doc_seg:
            self.doc_seg_clear = self.cut_doc()
        else:
            self.origin_doc_seg = self.doc
            self.doc_seg_clear = self._clear_seg_list(self.doc)
        self.doc_seg_list = [seg_info[0] for seg_info in self.doc_seg_clear]
        self.origin_doc_seg_list = [seg_info[0] for seg_info in self.origin_doc_seg]

    def cut_doc(self):
        """
        将文档进行分词处理
        :return:
        """
        logger.info('文档文本未分词，使用thunlp进行分词')
        self.thunlp_model = thulac.thulac(seg_only=True, model_path=THUNLP_MODEL_PATH, \
                                 user_dict=THUNLP_USER_DIC_PATH)
        doc_seg = self.thunlp_model.cut(self.doc)
        # 保存原始分词结果，进行关键相邻词的短语组合
        self.origin_doc_seg = doc_seg
        doc_seg_clear = self._clear_seg_list(doc_seg)
        logger.info('分词结束...')
        return doc_seg_clear

    def _clear_seg_list(self, doc_seg):
        """
        清洗分词结果，主要步骤为去除词性重要度不高的词，去除停用词，去除标点符号
        :param doc_seg: 初始的分词结果
        :return:
        """
        doc_seg_clear = self._filter_tag(doc_seg)
        doc_seg_clear = self._remove_stop_word(doc_seg_clear)
        doc_seg_clear = self._remove_punct(doc_seg_clear)
        return doc_seg_clear

    def _filter_tag(self, seg_list, tag_filter=['h', 'k', 'i', 'j', 'r', 'c', 'p',\
                                              'u', 'y', 'e', 'o', 'g', 'w', 'x']):
        """

        :param seg_list:
        :param tag_filter: 需要过滤的词性
         h/前接成分 k/后接成分 i/习语
        j/简称 r/代词 c/连词 p/介词 u/助词 y/语气助词
        e/叹词 o/拟声词 g/语素 w/标点 x/其它
        :return:
        """

        return [seg_info for seg_info in seg_list if seg_info[1] not in tag_filter]

    def _remove_stop_word(self, seg_list, stop_word_dic_path=STOP_WORD_DIC_PATH):
        """
        去除停用词
        :param seg_list:
        :param stop_word_dic_path: 停用词典文件路径
        :return:
        """
        with open(self.stop_word_dic_path, 'rb') as f:
            stop_word_list = f.read().split('\n')
        return [seg_info for seg_info in seg_list if seg_info[0] not in stop_word_list]

    def _remove_punct(self, seg_list, punct=punct):
        """
        去除常用标点和符号
        :param seg_list:
        :param punct:
        :return:
        """
        return [seg_info for seg_info in seg_list if seg_info[0] not in punct]

    def count_relation(self):
        """
        通过滑动窗口，统计词间联系数量
        :return:
        """
        word_count_dic = {}
        doc_length = len(self.doc_seg_list)
        if doc_length > self.window_size:
            for index in range(0, doc_length - self.window_size + 1):
                word = self.doc_seg_list[index]
                if word not in word_count_dic.keys():
                    word_count_dic[word] = list()
                else:
                    for seg in self.doc_seg_list[index+1: index+self.window_size]:
                        word_count_dic[word].append(seg)
        else:
            logger.warning('文档长度小于滑动窗口长度')
            pass
        return word_count_dic

    def build_graph(self):
        """

        :return:
        """
        word_length = len(set(self.doc_seg_list))
        matrix = np.zeros(word_length, word_length)
        word_count_dic = self.count_relation()
        word_index_dic = {}
        for index, word in enumerate(set(self.doc_seg_list)):
            word_index_dic[word] = index
        for word in word_index_dic.keys():
            pass