# coding=utf-8
"""
http://thulac.thunlp.org/
需要设置模型路径和用户字典路径
"""
import thulac

thu1 = thulac.thulac(seg_only=True, model_path="/home/showlove/cc/code/THULAC-Python/models",user_dict='/home/showlove/PycharmProjects/data_test/nlp/user_dic.txt')  #设置模式为行分词模式
a = thu1.cut('这是斯坦福中文分词器测试,三硝基对甲苯是一个测试测试词语,tfboys经常会玩儿会DOTA2')

for item in a:
    print item[0].decode('utf-8')
