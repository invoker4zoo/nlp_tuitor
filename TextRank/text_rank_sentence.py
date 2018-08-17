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

# GOLBAL PARAMS
THUNLP_MODEL_PATH = "/home/showlove/cc/code/THULAC-Python/models"
THUNLP_USER_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/user_dic.txt"
STOP_WORD_DIC_PATH = "/home/showlove/PycharmProjects/data_test/nlp/stop_word_dic.txt"
VECTOR_MODEL = "/home/showlove/cc/nlp/vector/sgns.renmin.bigram-char"
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
VECTOR_SIZE = 300

class TextSummary4Sentence(object):

    def __init__(self, doc):
        self.doc = doc
        self.vector_model = KeyedVectors.load_word2vec_format(VECTOR_MODEL)
        # seg_only 是否做词性分析
        self.thunlp_model = thulac.thulac(seg_only=False, model_path=THUNLP_MODEL_PATH, \
                                 user_dict=THUNLP_USER_DIC_PATH)

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
                matrix[i][j] = self.cal_sentence_similarity(self.doc_sentence_seg[i],\
                                                                           self.doc_sentence_seg[j])

                matrix[j][i] = matrix[i][j]
        return matrix

    def cal_importance(self):
        """

        :return:
        """
        self.matrix = self.build_graph()
        imp_matrix = np.ones([self.sentence_len, 1])
        for _ in range(0, step):
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