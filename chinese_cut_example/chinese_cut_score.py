# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: chinese_cut_score.py
@ time: $17-7-20 下午5:18
"""
import os
train_file_path_1 = '/home/showlove/cc/icwb2-data/gold/pku_training_words.utf8'
test_file_path_1 = '/home/showlove/cc/icwb2-data/gold/pku_test_gold.utf8'
train_file_path_2 = '/home/showlove/cc/icwb2-data/gold/msr_training_words.utf8'
test_file_path_2 = '/home/showlove/cc/icwb2-data/gold/msr_test_gold.utf8'
shell_path = '/home/showlove/cc/icwb2-data/scripts/score'
nltk_result_path_1 = '/home/showlove/PycharmProjects/data_test/tmp/nltk_testing/pku_outfile'
nltk_result_path_2 = '/home/showlove/PycharmProjects/data_test/tmp/nltk_testing/msr_outfile'
cmd = '{shell} {train_file} {test_file} {cut_path}'.format(shell=shell_path, train_file=train_file_path_2,
                                                                test_file=test_file_path_2, cut_path=nltk_result_path_2)
print cmd
print os.popen(cmd).read()