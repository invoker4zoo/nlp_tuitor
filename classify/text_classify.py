# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: text_classify.py
@ time: $17-7-18 下午7:15
"""
# coding=utf-8

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
CLASS_NUM = 6
SAMPLE_NUM = 4000

# init thunlp
THUNLP = thulac.thulac(seg_only=True, model_path=THUNLP_MODEL_PATH,user_dict=THUNLP_USER_DIC_PATH)

class loadFiles(object):
    """
    iter read sample file
    """
    def __init__(self,sample_path):
        self.sample_path = sample_path

    def load_file(self):
        if len(os.listdir(self.sample_path)):
            for file in os.listdir(self.sample_path):
                file_path = os.path.join(self.sample_path, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('utf-8')
                    tag = file.split('_')[0]
                    yield content, tag
        else:
            yield None, None


def convert_doc(doc_str):
    """
    读取文本，使用thunlp进行分词，对文本进行处理，生成可用文本列表
    :param doc_str:读入的单个文件的文本
    :return:
    """
    # 分词
    doc_str_list = get_thunlp_cut_list(THUNLP.cut(doc_str))
    # 除去停用词
    doc_str_list = rm_stop_word(doc_str_list)
    # 除去标点和特殊字符
    doc_str_list = rm_punct(doc_str_list)
    return doc_str_list

def rm_stop_word(doc_str_list):
    """
    删除停用词
    :param doc_str:
    :return:
    """
    with open(STOP_WORD_DIC_PATH, 'rb') as f:
        stop_word_list = f.read().split('\n')
    for n, item in enumerate(doc_str_list):
        if item in stop_word_list:
            doc_str_list.pop(n)
    return doc_str_list


def rm_punct(doc_str_list):
    """
    删除字符
    :param doc_str_list:
    :return:
    """
    for n, item in enumerate(doc_str_list):
        if item in punct or item == '\u300':
            doc_str_list.pop(n)
    return doc_str_list


def get_thunlp_cut_list(thunlp_cut):
    """
    将thunlp分词结果提取为字符列表
    :param thunlp_cut:
    :return:
    """
    result_list = list()
    for item in thunlp_cut:
        result_list.append(item[0])
    return result_list


def svm_classify(train_set,train_tag,test_set,test_tag):
    """
    使用sklearnsvm方法
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

    train_err_num, train_err_ratio = checkPred(train_tag, train_pred)
    test_err_num, test_err_ratio  = checkPred(test_tag, test_pred)

    print('=== 分类训练完毕，分类结果如下 ===')
    print('训练集误差: {e}'.format(e=train_err_ratio))
    print('检验集误差: {e}'.format(e=test_err_ratio))

    return clf_res


def checkPred(data_tag, data_pred):
    if len(data_tag) != len(data_pred):
        raise RuntimeError('The length of data tag and data pred should be the same')
    err_count = 0
    for i in range(len(data_tag)):
        if data_tag[i]!=data_pred[i]:
            err_count += 1
    err_ratio = err_count / len(data_tag)
    return [err_count, err_ratio]


def text_classify(text, dictionary=None, lsi_model=None, predictor=None):
    """
    文本分类，字符串
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
    if not dictionary:
        dictionary = corpora.Dictionary.load(DICTIONARY_PATH)
    if not lsi_model:
        with open(LSI_MODEL, 'rb') as f:
            lsi_model = pickle.load(f)
    if not predictor:
        with open(PREDICTOR_MODEL, 'rb') as f:
            predictor = pickle.load(f)
    files = os.listdir(LSI_PATH)
    tags_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in tags_list:
            tags_list.append(t)

    text_str = convert_doc(text)
    text_bow = dictionary.doc2bow(text_str)
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    text_tfidf = tfidf_model[text_bow]
    text_lsi = lsi_model[text_tfidf]
    data = list()
    cols = list()
    rows = list()
    for item in text_lsi:
        data.append(item[1])
        rows.append(0)
        cols.append(item[0])
    text_matrix = csr_matrix((data, (rows, cols))).toarray()
    predict_tag_index = predictor.predict(text_matrix)
    return tag_dic[tags_list[predict_tag_index[0]]]

if __name__ == '__main__':
    dictionary = None
    lsi_model = None
    predictor = None
    # 生成字典
    # #############################################################
    if not os.path.exists(DICTIONARY_PATH):
        print '*' * 20 + '未检测到词典,生成词典中' + '*' * 20
        files = loadFiles(TEXT_SAMPLE_DIR).load_file()
        dictionary = corpora.Dictionary()
        # loading example file
        for i, file in enumerate(files):
            if file[0]:
                doc_str = file[0]
                # reading file is u-code trans to utf-8
                doc_str = doc_str.encode('utf-8')
                tag = file[1]
                doc_str_list = convert_doc(doc_str)
                dictionary.add_documents([doc_str_list])
                if i%100==0:
                    print '[%s] %d file has been loaded'%\
                          (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()),i)



            else:
                print '[error] example dir is not exit'
                sys.exit()
        # 去掉词频过低的词
        low_freq_ids = [tokenid for tokenid, freq in dictionary.dfs.items() if freq < 3]
        dictionary.filter_tokens(low_freq_ids)
        dictionary.compactify()
        dictionary.save(DICTIONARY_PATH)
        print '*' * 20 + '已生成词典' + '*' * 20
    else:
        print '*' * 20 + '已检测到词典,读入中' + '*' * 20
        dictionary = corpora.Dictionary.load(DICTIONARY_PATH)
        print '*' * 20 + '字典已读入' + '*' * 20

    # 文档tfidf化
    # #################################################################
    if not os.path.exists(TFIDF_PATH):
        print '*' * 20 + '未检测tf-idf文件,生成tfidf文件' + '*' * 20
        os.mkdir(TFIDF_PATH)
        files = loadFiles(TEXT_SAMPLE_DIR).load_file()
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        corpus_tfidf = dict()
        # loading example file
        for i, file in enumerate(files):
            if file[0]:
                doc_str = file[0]
                # reading file is u-code trans to utf-8
                doc_str = doc_str.encode('utf-8')
                tag = file[1]
                doc_str_list = convert_doc(doc_str)
                doc_bow = dictionary.doc2bow(doc_str_list)
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
            saving_path = TFIDF_PATH + '/' + tag + '.mm'
            corpora.MmCorpus.serialize(saving_path, corpus_tfidf.get(tag), id2word=dictionary)
            print 'tag %s has been transformed into tfidf vector'%tag

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


    # 文档LSI向量生成
    # #################################################################
    if not os.path.exists(LSI_PATH):
        print '*' * 20 + '未检测LSI文件夹,生成LSI文件' + '*' * 20
        os.mkdir(LSI_PATH)
        corpus_tfidf_all = list()
        tags_list = corpus_tfidf.keys()
        for tag in tags_list:
            tmp = corpus_tfidf.get(tag, [])
            corpus_tfidf_all += tmp
        # num_topics 为svd分解取的特征数量
        lsi_model = models.LsiModel(corpus_tfidf_all, id2word=dictionary, num_topics=50)
        with open(LSI_MODEL, 'wb') as f:
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


    # classify predictor produce
    # 分类器生成
    # #################################################################
    if not os.path.exists(PREDICTOR_MODEL):
        print '*' * 20 + '未检测分类模型文件，生成中' + '*' * 20
        tag_list = list()
        tag_index_list = list()
        doc_num_list = list()
        corpus_lsi_all = list()
        files = os.listdir(LSI_PATH)
        for file in files:
            t = file.split('.')[0]
            if t not in tag_list:
                tag_list.append(t)
        for i,tag in enumerate(tag_list):
            tmp = corpus_lsi[tag]
            tag_index_list += [i] * len(tmp)
            doc_num_list.append(len(tmp))
            corpus_lsi_all += tmp

        # gensim mm 文件转换为numpy形式，进行分类器训练
        data = list()
        rows = list()
        cols = list()
        line_count = 0
        for line in corpus_lsi_all:
            for item in line:
                rows.append(line_count)
                cols.append(item[0])
                data.append(item[1])
            line_count += 1
        lsi_matrix = csr_matrix((data, (rows, cols))).toarray()
        # 生成训练集和测试集
        rarray = np.random.random(size=line_count)
        train_set = []
        train_tag = []
        test_set = []
        test_tag = []
        for i in range(line_count):
            if rarray[i] < 0.8:
                train_set.append(lsi_matrix[i, :])
                train_tag.append(tag_index_list[i])
            else:
                test_set.append(lsi_matrix[i, :])
                test_tag.append(tag_index_list[i])

        # 生成分类器
        predictor = svm_classify(train_set, train_tag, test_set, test_tag)
        # 分类器模型存储
        with open(PREDICTOR_MODEL,'wb') as f:
            pickle.dump(predictor, f)
        print '*' * 20 + '分类模型文件已生成' + '*' * 20

    else:
        print '*' * 20 + '已检测到分类模型文件' + '*' * 20



    # 文本测试
    if not dictionary:
        dictionary = corpora.Dictionary.load(DICTIONARY_PATH)
    if not lsi_model:
        with open(LSI_MODEL, 'rb') as f:
            lsi_model = pickle.load(f)
    if not predictor:
        with open(PREDICTOR_MODEL, 'rb') as f:
            predictor = pickle.load(f)
    files = os.listdir(LSI_PATH)
    tags_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in tags_list:
            tags_list.append(t)
    # demo_doc = """
    # 这次大选让两党的精英都摸不着头脑。以媒体专家的传统观点来看，要选总统首先要避免失言，避免说出一些“offensive”的话。希拉里，罗姆尼，都是按这个方法操作的。罗姆尼上次的47%言论是在一个私人场合被偷录下来的，不是他有意公开发表的。今年希拉里更是从来没有召开过新闻发布会。
    # 川普这种肆无忌惮的发言方式，在传统观点看来等于自杀。
    # """
    demo_doc = """
        网易体育2月11日讯：^M
    2007/2008赛季CBA联赛总决赛首回合比赛^M
    将于北京时间2月13日晚7点半正式打响^M
    ，首场较量华南虎广东宏远将坐镇主场迎接东北虎辽宁盼盼的挑战，比赛打到这个份上，总
    冠军奖杯近在咫尺，谁都不想遗憾地错过，本轮比赛，两只老虎势必会有一场殊死之战。^M
    相对于篮球场上其它位置，大前锋在队上担任的任务几乎都是以苦工为主，要抢篮板、防守
    、卡位都少不了他，但是要投篮、得分，他却经常是最后一个，从一定程度上说，大前锋是
    篮球场上最不起眼的。但是就是这个位置，却往往在比赛中扮演着至关重要的角色。下面就
    让我们来比较以下两队在这个位置上的人员配置。^M
    广东队这个位置杜锋、朱芳雨都能独挡一面，即使在国内篮坛来说，这个人员储备都称得上
    是豪华。辽宁队的刘相韬、李晓旭与谷立业就队内来说也是这个位置上的好手。但是把他们
    放到一个同等的界面上来说，却又有很大的不同。^M
    国内名气方面：^M
    广东队无疑要远远胜于辽宁，无论是杜锋还是朱芳雨都是国字号球员，在国内篮坛都是赫赫
    有名的角色，相比较而言，辽宁队的刘相韬，谷立业尽管在辽宁上有一些名气，但是在国内
    篮坛他们还远称不上“大腕”。^M
    个人技术方面：
    """
    demo_doc = """
    　红魔夏季热身赛第二场，曼联2-1战胜皇家盐湖城。虽然友谊赛的含金量不算高，但回顾两场比赛，姆希塔良的表现都可圈可点。他已经逐渐奠定了自己在攻击线上不可获取的地位。此役对阵皇家盐湖城，姆希塔良一传一射。更令穆里尼奥欣喜的是，他与新援卢卡库之间正在形成不错的化学反应。
    """


    print '分类结果为: %s'%text_classify(demo_doc)
