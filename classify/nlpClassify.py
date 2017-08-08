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
from data.punct import punct
from gensim import corpora, models
import time
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

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
# CLASS_NUM = 6
# SAMPLE_NUM = 4000


class nlpClassifier(object):

    def __init__(self, thunlp_model_path=THUNLP_MODEL_PATH,
                    thunlp_user_dic_path=THUNLP_USER_DIC_PATH,
                    text_example_path=TEXT_SAMPLE_DIR,
                    stop_word_dic_path=STOP_WORD_DIC_PATH,
                    saving_word_dictionary_path=DICTIONARY_PATH,
                    saving_tfidf_path=TFIDF_PATH,
                    saving_lsi_file_path=LSI_PATH,
                    saving_lsi_model_path=LSI_MODEL,
                    saving_predictor_model_path=PREDICTOR_MODEL

                 ):

        self.thunlp_model_path = thunlp_model_path
        self.thunlp_user_dic_path = thunlp_user_dic_path
        self.text_example_path = text_example_path
        self.stop_word_dic_path = stop_word_dic_path
        self.saving_word_dictionary_path = saving_word_dictionary_path
        self.saving_tfidf_path = saving_tfidf_path
        self.saving_lsi_ftile_path = saving_lsi_file_path
        self.saving_lsi_model_path = saving_lsi_model_path
        self.saving_predictor_model_path = saving_predictor_model_path
        self.THUNLP = self._init_thunlp()
        self.train()

    def _init_needed_file(self):
        """
        需要的文件：dictionary, tf-idf file, lsi file, predictor model
        通过初始化地址去找需要的文件，如果不存在文件则进行生成
        :return:
        """
        if not self.dictionary:
            self.dictionary = corpora.Dictionary.load(self.saving_word_dictionary_path)
        if not self.lsi_model:
            with open(self.saving_lsi_model_path, 'rb') as f:
                self.lsi_model = pickle.load(f)
        if not self.predictor:
            with open(self.saving_predictor_model_path, 'rb') as f:
                self.predictor = pickle.load(f)

    def train(self):
        """
        从样本文本中进行训练，生成dictionary, tf-idf file, lsi file, predictor model
        :return:
        """
        self.build_text_dictionary()
        self.build_tfidf_file()
        self.build_lsi_file()

    def _init_thunlp(self):
        """
        初始化thulac模型
        :return:
        """
        return thulac.thulac(seg_only=True, model_path=self.thunlp_model_path,user_dict=self.thunlp_user_dic_path)

    def get_tag_list(self):
        """
        从样本文本中得到标签列表
        :return:
        """

        files = os.listdir(self.saving_lsi_file_path)
        tags_list = []
        for file in files:
            t = file.split('.')[0]
            if t not in tags_list:
                tags_list.append(t)
        return tags_list

    def _loading_example_text(self):
        """
        得到模型训练文本文件的迭代器
        :return: iteration yield
        """
        if len(os.listdir(self.text_example_path)):
            for file in os.listdir(self.text_example_path):
                file_path = os.path.join(self.text_example_path, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('utf-8')
                    tag = file.split('_')[0]
                    yield content, tag
        else:
            yield None, None

    def convert_doc(self, doc_str):
        """
        读取文本，使用thunlp进行分词，对文本进行处理，生成可用文本列表
        :param doc_str:读入的单个文件的文本
        :return:
        """
        # 分词
        doc_str_list = self.get_thunlp_cut_list(self.THUNLP.cut(doc_str))
        # 除去停用词
        doc_str_list = self.rm_stop_word(doc_str_list)
        # 除去标点和特殊字符
        doc_str_list = self.rm_punct(doc_str_list)
        return doc_str_list

    def get_thunlp_cut_list(self, thunlp_cut):
        """
        将thunlp分词结果提取为字符列表
        :param thunlp_cut:
        :return:
        """
        result_list = list()
        for item in thunlp_cut:
            result_list.append(item[0])
        return result_list

    def rm_stop_word(self, doc_str_list):
        """
        删除停用词
        :param doc_str:
        :return:
        """
        with open(self.stop_word_dic_path, 'rb') as f:
            stop_word_list = f.read().split('\n')
        for n, item in enumerate(doc_str_list):
            if item in stop_word_list:
                doc_str_list.pop(n)
        return doc_str_list

    def rm_punct(self, doc_str_list):
        """
        删除字符
        :param doc_str_list:
        :return:
        """
        for n, item in enumerate(doc_str_list):
            if item in punct or item == '\u300':
                doc_str_list.pop(n)
        return doc_str_list

    def svm_classify(self, train_set, train_tag, test_set,test_tag):
        """
        使用sklearnsvm方法,训练模型，返回模型类
        :param train_set:
        :param train_tag:
        :param test_set:
        :param test_tag:
        :return:
        """
        clf = svm.LinearSVC()
        clf_res = clf.fit(train_set,train_tag)
        train_pred  = clf_res.predict(train_set)
        test_pred = clf_res.predict(test_set)

        train_err_num, train_err_ratio = self.checkPred(train_tag, train_pred)
        test_err_num, test_err_ratio  = self.checkPred(test_tag, test_pred)

        print('=== 分类训练完毕，分类结果如下 ===')
        print('训练集误差: {e}'.format(e=train_err_ratio))
        print('检验集误差: {e}'.format(e=test_err_ratio))

        return clf_res

    def checkPred(self, data_tag, data_pred):
        if len(data_tag) != len(data_pred):
            raise RuntimeError('The length of data tag and data pred should be the same')
        err_count = 0
        for i in range(len(data_tag)):
            if data_tag[i]!=data_pred[i]:
                err_count += 1
        err_ratio = err_count / len(data_tag)
        return [err_count, err_ratio]

    def build_text_dictionary(self):
        """
        将训练文本集中所有文本分词后，形成词袋文件
        词典中词具有word id
        使用gensim库中的corpora模块
        :return:
        """
        if not os.path.exists(DICTIONARY_PATH):
            print '*' * 20 + '未检测到词典,生成词典中' + '*' * 20
            files = self._loading_example_text()
            dictionary = corpora.Dictionary()
            # loading example file
            for i, file in enumerate(files):
                if file[0]:
                    doc_str = file[0]
                    # reading file is u-code trans to utf-8
                    doc_str = doc_str.encode('utf-8')
                    tag = file[1]
                    doc_str_list = self.convert_doc(doc_str)
                    dictionary.add_documents([doc_str_list])
                    if i % 100 == 0:
                        print '[%s] %d file has been loaded' % \
                              (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), i)

                else:
                    print '[error] example dir is not exist'
                    sys.exit()
            # 去掉词频过低的词
            low_freq_ids = [tokenid for tokenid, freq in dictionary.dfs.items() if freq < 3]
            dictionary.filter_tokens(low_freq_ids)
            dictionary.compactify()
            dictionary.save(self.saving_word_dictionary_path)
            self.dictionary = dictionary
            print '*' * 20 + '已生成词典' + '*' * 20
        else:
            print '*' * 20 + '已检测到词典,读入中' + '*' * 20
            self.dictionary = corpora.Dictionary.load(DICTIONARY_PATH)
            print '*' * 20 + '字典已读入' + '*' * 20

    def build_tfidf_file(self):
        """
        生成tfidf文件,tf-idf文件是为了作为生成lsi文件的基础
        不会直接运用在模型生成中
        :return:
        """
        if not os.path.exists(self.saving_tfidf_path):
            print '*' * 20 + '未检测tf-idf文件,生成tfidf文件' + '*' * 20
            os.mkdir(self.saving_tfidf_path)
            files = self._loading_example_text()
            tfidf_model = models.TfidfModel(dictionary=self.dictionary)
            corpus_tfidf = dict()
            # loading example file
            for i, file in enumerate(files):
                if file[0]:
                    doc_str = file[0]
                    # reading file is u-code trans to utf-8
                    doc_str = doc_str.encode('utf-8')
                    tag = file[1]
                    doc_str_list = self.convert_doc(doc_str)
                    doc_bow = self.dictionary.doc2bow(doc_str_list)
                    doc_tfidf = tfidf_model[doc_bow]
                    tmp = corpus_tfidf.get(tag, [])
                    tmp.append(doc_tfidf)
                    if len(tmp) == 1:
                        corpus_tfidf[tag] = tmp
                    if i % 100 == 0:
                        print '[%s] %d file has been transformed into tfidf file' \
                              % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), i)
                else:
                    print '[error] example dir is not exit'
                    sys.exit()

            # store tfidf file
            tags = corpus_tfidf.keys()
            for tag in tags:
                saving_path = self.saving_tfidf_path + '/' + tag + '.mm'
                corpora.MmCorpus.serialize(saving_path, corpus_tfidf.get(tag), id2word=self.dictionary)
                print 'tag %s has been transformed into tfidf vector' % tag

            print '*' * 20 + '已生成tfidf分类文件' + '*' * 20
        else:
            print '*' * 20 + '已检测tfidf文件夹' + '*' * 20
            files = os.listdir(TFIDF_PATH)
            tags_list = list()
            for file in files:
                t = file.split('.')[0]
                if t not in tags_list:
                    tags_list.append(t)
            corpus_tfidf = dict()
            for tag in tags_list:
                saving_path = TFIDF_PATH + '/' + tag + '.mm'
                corpus = corpora.MmCorpus(saving_path)
                corpus_tfidf[tag] = corpus
            print '*' * 20 + 'tfidf文件已读入' + '*' * 20

    def build_lsi_file(self):
        """
        创建lsi向量文件
        :return:
        """
        if not os.path.exists(self.saving_lsi_file_path):
            print '*' * 20 + '未检测LSI文件夹,生成LSI文件' + '*' * 20
            os.mkdir(self.saving_lsi_file_path)
            corpus_tfidf_all = list()
            tags_list = corpus_tfidf.keys()
            for tag in tags_list:
                tmp = corpus_tfidf.get(tag, [])
                corpus_tfidf_all += tmp
            # num_topics 为svd分解取的特征数量
            lsi_model = models.LsiModel(corpus_tfidf_all, id2word=dictionary, num_topics=50)
            with open(self.saving_lsi_model_path, 'wb') as f:
                pickle.dump(lsi_model, f)
            print '*' * 20 + 'LSI模型文件已生成' + '*' * 20
            # 释放空间
            del corpus_tfidf_all
            # 生成lsi向量文件
            corpus_lsi = dict()
            for tag in tags_list:
                corpu = [lsi_model[doc] for doc in corpus_tfidf.get(tag)]
                corpus_lsi[tag] = corpu
                saving_path = LSI_PATH + '/' + tag + '.mm'
                corpora.MmCorpus.serialize(saving_path, corpu, id2word=dictionary)
            print '*' * 20 + 'LSI向量文件已生成' + '*' * 20
        else:
            print '*' * 20 + '已检测LSI文件夹' + '*' * 20
            files = os.listdir(LSI_PATH)
            tags_list = list()
            for file in files:
                t = file.split('.')[0]
                if t not in tags_list:
                    tags_list.append(t)
            corpus_lsi = dict()
            for tag in tags_list:
                saving_path = LSI_PATH + '/' + tag + '.mm'
                corpus = corpora.MmCorpus(saving_path)
                corpus_lsi[tag] = corpus
            print '*' * 20 + 'LSI向量文件已读入' + '*' * 20

    def text_classify(self, text):
        """
        文本分类主函数
        需要读入字典文件,lsi_model文件,和模型predictor 文件
        :param text:
        :return:
        """
        tag_dic = {
            'Military': '军事',
            'Culture': '文化',
            'Auto': '汽车',
            'Sports': '体育',
            'Economy': '经济',
            'Medicine': '医药'
        }
        self._init_needed_file()
        # if not self.dictionary:
        #     self.dictionary = corpora.Dictionary.load(self.saving_word_dictionary_path)
        # if not self.lsi_model:
        #     with open(self.saving_lsi_model_path, 'rb') as f:
        #         self.lsi_model = pickle.load(f)
        # if not self.predictor:
        #     with open(self.saving_predictor_model_path, 'rb') as f:
        #         self.predictor = pickle.load(f)

        tags_list = self.get_tag_list()


        text_str = self.convert_doc(text)
        text_bow = self.dictionary.doc2bow(text_str)
        tfidf_model = models.TfidfModel(dictionary=self.dictionary)
        text_tfidf = tfidf_model[text_bow]
        text_lsi = self.lsi_model[text_tfidf]
        data = list()
        cols = list()
        rows = list()
        for item in text_lsi:
            data.append(item[1])
            rows.append(0)
            cols.append(item[0])
        text_matrix = csr_matrix((data, (rows, cols))).toarray()
        predict_tag_index = self.predictor.predict(text_matrix)
        return tag_dic[tags_list[predict_tag_index[0]]]