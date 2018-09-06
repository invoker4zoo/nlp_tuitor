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
from gensim.models import KeyedVectors
import math

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# GOLBAL PARAMS
THUNLP_MODEL_PATH = "/home/showlove/cc/code/THULAC-Python/models"
THUNLP_USER_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/user_dic.txt"
STOP_WORD_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/stop_word_dic.txt"
VECTOR_MODEL = "/home/showlove/cc/nlp/vector/sgns.renmin.bigram-char"
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…']
VECTOR_SIZE = 300

class TextSummary4Sentence(object):

    def __init__(self, doc, step, alpha):
        self.doc = doc
        self.step = step
        self.alpha = alpha
        # self.vector_model = KeyedVectors.load_word2vec_format(VECTOR_MODEL)
        # seg_only 是否做词性分析
        self.thunlp_model = thulac.thulac(seg_only=False, model_path=THUNLP_MODEL_PATH, \
                                 user_dict=THUNLP_USER_DIC_PATH)
        self.cut_sentence()
        self.cut_sentence_seg()
        self.cal_importance()

    def cut_sentence(self):
        """
        文档的分句
        :return:
        """
        self.doc_sentence = []
        sentnce = ''
        for char in self.doc:
            if char not in sentence_delimiters:
                sentnce += char
            else:
                self.doc_sentence.append(sentnce)
                sentnce = ''

    def cut_sentence_seg(self):
        """
        句子的分词，用于计算句子的向量
        :return:
        """
        self.doc_sentence_seg = []
        for sentence in self.doc_sentence:
            sentence_seg = self.thunlp_model.cut(sentence)
            sentence_seg_clear = self._clear_seg_list(sentence_seg)
            self.doc_sentence_seg.append(sentence_seg_clear)

    def cos_distance(self,vector1, vector2):
        """

        :param vector1:
        :param vector2:
        :return:
        """
        tx = np.array(vector1)
        ty = np.array(vector2)
        cos1 = np.sum(tx * ty)
        cos21 = np.sqrt(sum(tx ** 2))
        cos22 = np.sqrt(sum(ty ** 2))
        cosine_value = cos1 / float(cos21 * cos22)
        return cosine_value

    def two_sentences_similarity(self, sents_1, sents_2):
        '''
        计算两个句子的相似性
        :param sents_1:
        :param sents_2:
        :return:
        '''
        counter = 0
        for sent in sents_1:
            if sent in sents_2:
                counter += 1
        return counter / (math.log(len(sents_1) + len(sents_2)))

    def cal_sentence_similarity(self, sentence1, sentence2):
        """

        :param sentence1:
        :param sentence2:
        :return:
        """
        if len(sentence1) == 0 or len(sentence2) == 0:
            return 0.0
        for index,seg in enumerate(sentence1):
            if index == 0:
                vector1 = self.vector_model[seg]
            else:
                vector1 += self.vector_model[seg]

        for index, seg in enumerate(sentence2):
            if index == 0:
                vector2 = self.vector_model[seg]
            else:
                vector2 += self.vector_model[seg]
        similarity = self.cos_distance(vector1/len(sentence1), vector2/len(sentence2))
        return similarity

    def build_graph(self):
        """
        建立句子链表的关系图
        :return:
        """
        sentence_len = len(self.doc_sentence_seg)
        self.sentence_len = sentence_len
        matrix = np.zeros([sentence_len, sentence_len])
        for i in range(0, sentence_len):
            for j in range(i, sentence_len):
                if i==j:
                    matrix[i][j] = 1
                    continue
                # matrix[i][j] = self.cal_sentence_similarity(self.doc_sentence_seg[i],\
                #                                                            self.doc_sentence_seg[j])
                matrix[i][j] = self.two_sentences_similarity(self.doc_sentence_seg[i],\
                                                                           self.doc_sentence_seg[j])

                matrix[j][i] = matrix[i][j]
        # 归一化
        for j in range(matrix.shape[1]):
            sum = 0
            for i in range(matrix.shape[0]):
                sum += matrix[i][j]
            for i in range(matrix.shape[0]):
                matrix[i][j] /= sum
        return matrix

    def cal_importance(self):
        """

        :return:
        """
        self.matrix = self.build_graph()
        imp_matrix = np.ones([self.sentence_len, 1])
        for _ in range(0, self.step):
            imp_matrix_hat = (1 - self.alpha) + self.alpha * np.dot(self.matrix, imp_matrix)
            # 判断终止条件
            ###########
            imp_matrix = imp_matrix_hat
        self.imp_matrix = imp_matrix

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

    def _filter_tag(self, seg_list, tag_filter=['a', 'd', 'v', 'n', 'ns', 'ni', 'vm', 'vd'], reverse=False):
        """

        :param seg_list:
        :param tag_filter: 需要过滤的词性
        n/名词 np/人名 ns/地名 ni/机构名 nz/其它专名
        m/数词 q/量词 mq/数量词 t/时间词 f/方位词 s/处所词
        v/动词 a/形容词 d/副词 h/前接成分 k/后接成分 i/习语
        j/简称 r/代词 c/连词 p/介词 u/助词 y/语气助词
        e/叹词 o/拟声词 g/语素 w/标点 x/其它 vm/能愿动词 vd/趋向动词
        :return:
        """
        if reverse:
            return [seg_info for seg_info in seg_list if seg_info[1] not in tag_filter]
        else:
            return [seg_info for seg_info in seg_list if seg_info[1] in tag_filter]

    def _remove_stop_word(self, seg_list, stop_word_dic_path=STOP_WORD_DIC_PATH):
        """
        去除停用词
        :param seg_list:
        :param stop_word_dic_path: 停用词典文件路径
        :return:
        """
        with open(stop_word_dic_path, 'rb') as f:
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

    def print_result(self, top_n=5):
        """

        :param top_n:
        :return:
        """
        word_imp = {}
        for index in range(0, len(self.imp_matrix)):
            word_imp[self.doc_sentence[index]] = self.imp_matrix[index][0]
        result = sorted(word_imp.items(), key=lambda x:x[1], reverse=True)
        for item in result[0:top_n]:
            print item[0] + ':' + str(item[1])


if __name__ == '__main__':
    # doc = u'程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。'
    doc = u"""
        网易体育2月11日讯：2007/2008赛季CBA联赛总决赛首回合比赛将于北京时间2月13日晚7点半正式打响，首场较量华南虎广东宏远将坐镇主场迎接东北虎辽宁盼盼的挑战，比赛打到这个份上，总
    冠军奖杯近在咫尺，谁都不想遗憾地错过，本轮比赛，两只老虎势必会有一场殊死之战。
    相对于篮球场上其它位置，大前锋在队上担任的任务几乎都是以苦工为主，要抢篮板、防守
    、卡位都少不了他，但是要投篮、得分，他却经常是最后一个，从一定程度上说，大前锋是
    篮球场上最不起眼的。但是就是这个位置，却往往在比赛中扮演着至关重要的角色。下面就
    让我们来比较以下两队在这个位置上的人员配置。
    广东队这个位置杜锋、朱芳雨都能独挡一面，即使在国内篮坛来说，这个人员储备都称得上
    是豪华。辽宁队的刘相韬、李晓旭与谷立业就队内来说也是这个位置上的好手。但是把他们
    放到一个同等的界面上来说，却又有很大的不同。
    国内名气方面：
    广东队无疑要远远胜于辽宁，无论是杜锋还是朱芳雨都是国字号球员，在国内篮坛都是赫赫
    有名的角色，相比较而言，辽宁队的刘相韬，谷立业尽管在辽宁上有一些名气，但是在国内
    篮坛他们还远称不上“大腕”。
    个人技术方面：
    """
    model = TextSummary4Sentence(doc, 700, 0.85)
    model.print_result()