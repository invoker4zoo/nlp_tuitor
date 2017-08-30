# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: data_util_zhihu.py
@ time: $17-8-30 上午10:31
"""
import codecs
import numpy as np
import word2vec
import os
import pickle
from tflearn.data_utils import pad_sequences

PAD_ID = 0
_GO="_GO"
_END="_END"
_PAD="_PAD"

def create_voabulary(simple=None,word2vec_model_path='../tmp/zhihu-word2vec-title-desc.bin-100',name_scope=''): #zhihu-word2vec-multilabel.bin-100
    cache_path ='cache_vocabulary_label_pik/'+ name_scope + "_word_voabulary.pik"
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with open(cache_path, 'r') as data_f:
            vocabulary_word2index, vocabulary_index2word=pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index={}
        vocabulary_index2word={}
        if simple is not None:
            word2vec_model_path='zhihu-word2vec.bin-100'
        print("create vocabulary. word2vec_model_path:",word2vec_model_path)
        model=word2vec.load(word2vec_model_path,kind='bin')
        vocabulary_word2index['PAD_ID']=0
        vocabulary_index2word[0]='PAD_ID'
        special_index=0
        # a special token for biLstTextRelation model. which is used between two sentences.
        if 'biLstmTextRelation' in name_scope:
            vocabulary_word2index['EOS']=1
            vocabulary_index2word[1]='EOS'
            special_index=1
        for i,vocab in enumerate(model.vocab):
            vocabulary_word2index[vocab]=i+1+special_index
            vocabulary_index2word[i+1+special_index]=vocab

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'a') as data_f:
                pickle.dump((vocabulary_word2index,vocabulary_index2word), data_f)
    return vocabulary_word2index,vocabulary_index2word


def create_voabulary_label(voabulary_label='../tmp/train-zhihu4-only-title-all.txt',name_scope='',use_seq2seq=False):#'train-zhihu.txt'
    """

    :param voabulary_label:
    :param name_scope:
    :param use_seq2seq:
    :return:
    """
    print("create_voabulary_label_sorted.started.traning_data_path:",voabulary_label)
    cache_path ='cache_vocabulary_label_pik/'+ name_scope + "_label_voabulary.pik"
    if os.path.exists(cache_path):#如果缓存文件存在，则直接读取
        with open(cache_path, 'r') as data_f:
            vocabulary_word2index_label, vocabulary_index2word_label=pickle.load(data_f)
            return vocabulary_word2index_label, vocabulary_index2word_label
    else:
        zhihu_f_train = codecs.open(voabulary_label, 'r', 'utf8')
        lines=zhihu_f_train.readlines()
        count=0
        vocabulary_word2index_label={}
        vocabulary_index2word_label={}
        vocabulary_label_count_dict={} #{label:count}
        for i,line in enumerate(lines):
            if '__label__' in line:  #'__label__-2051131023989903826
                label=line[line.index('__label__')+len('__label__'):].strip().replace("\n","")
                if vocabulary_label_count_dict.get(label,None) is not None:
                    vocabulary_label_count_dict[label]=vocabulary_label_count_dict[label]+1
                else:
                    vocabulary_label_count_dict[label]=1
        list_label=sort_by_value(vocabulary_label_count_dict)

        print("length of list_label:",len(list_label));#print(";list_label:",list_label)
        countt=0

        ##########################################################################################
        if use_seq2seq:#if used for seq2seq model,insert two special label(token):_GO AND _END
            i_list=[0,1,2]
            label_special_list=[_GO,_END,_PAD]
            for i,label in zip(i_list,label_special_list):
                vocabulary_word2index_label[label] = i
                vocabulary_index2word_label[i] = label
        #########################################################################################
        for i,label in enumerate(list_label):
            if i<10:
                count_value=vocabulary_label_count_dict[label]
                print("label:",label,"count_value:",count_value)
                countt=countt+count_value
            indexx = i + 3 if use_seq2seq else i
            vocabulary_word2index_label[label]=indexx
            vocabulary_index2word_label[indexx]=label
        print("count top10:",countt)

        #save to file system if vocabulary of words is not exists.
        if not os.path.exists(cache_path): #如果不存在写到缓存文件中
            with open(cache_path, 'a') as data_f:
                pickle.dump((vocabulary_word2index_label,vocabulary_index2word_label), data_f)
    print("create_voabulary_label_sorted.ended.len of vocabulary_label:",len(vocabulary_index2word_label))
    return vocabulary_word2index_label,vocabulary_index2word_label


def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [ backitems[i][1] for i in range(0,len(backitems))]


def load_data(vocabulary_word2index,vocabulary_word2index_label,valid_portion=0.05,max_training_data=1000000,training_data_path='../tmp/train-zhihu4-only-title-all.txt'):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645 w6 w111 __label__-400525901828896492"
    print("load_data.started...")
    zhihu_f = codecs.open(training_data_path, 'r', 'utf8') #-zhihu4-only-title.txt
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    for i, line in enumerate(lines):
        x, y = line.split('__label__') #x='w17314 w5521 w7729 w767 w10147 w111'
        y=y.replace('\n','')
        x = x.replace("\t",' EOS ').strip()
        if i<5:
            print("x0:",x) #get raw x
        #x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        #if i<5:
        #    print("x1:",x_) #
        x=x.split(" ")
        x = [vocabulary_word2index.get(e,0) for e in x] #if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        if i<5:
            print("x1:",x) #word to index
        y = vocabulary_word2index_label[y] #np.abs(hash(y))
        X.append(x)
        Y.append(y)
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:",number_examples) #
    train = (X[0:int((1 - valid_portion) * number_examples)], Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])
    # 5.return
    print("load_data.ended...")
    return train, test, test