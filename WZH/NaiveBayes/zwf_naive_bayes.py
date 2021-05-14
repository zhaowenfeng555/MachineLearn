"""
@ date: 20201028
@ desc: KMeans
@author: zhaowenfeng@163.com
"""

import math
import random
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes(object):
    """
    naive bayes
    """
    def __init__(self, data, label):
        """
        init
        """
        # data 是二维数组
        self.data = data
        self.dict_word2id = dict()
        self.label = label

    def get_dict_data(self):
        """
        get dict_vacabulary
        """
        counter = Counter()
        for corpus in self.data:
            counter += Counter(corpus)
        dict_counter = dict(counter)
        # print (dict_counter)
        list_order_dict_counter = sorted(dict_counter.items(), key = lambda x: x[1], reverse = True)
        print (list_order_dict_counter)
        for index, itm in enumerate(list_order_dict_counter):
            self.dict_word2id[itm[0]] = index
        return self.dict_word2id

    def list_word_to_vector(self, list_word):
        """
        word to vector
        """
        list_word_index = [0] * len(self.dict_word2id)
        for word in list_word:
            if word not in self.dict_word2id:
                print('error oov', str(word))
                continue
            else:
                list_word_index[self.dict_word2id.get(word, 99999)] = 1
        # print ('word', str(list_word))
        # print (str(list_word_index))
        return list_word_index

    def train(self, word2vector):
        """
        train
        """
        classify = MultinomialNB().fit(word2vector, self.label)
        return classify

    def infer(self, classify, list_infer):
        """
        infer
        """
        result = classify.predict_proba(list_infer)
        print (result)
        print (classify.predict(list_infer))

        # if infer_0 > infer_1:
        #     print ('属于 0 ******', str(infer_0), str(infer_1))
        # else:
        #     print ('属于 1 ******', str(infer_0), str(infer_1))


    def main(self):
        """
        main
        """
        dict_word2id = self.get_dict_data()
        print(str(dict_word2id))
        word2vector = []
        for corpus in self.data:
            word2vector.append(self.list_word_to_vector(corpus))
        print ('\nword2vector..', word2vector)


        classify = self.train(word2vector)

        data_infer = ['love', 'my', 'dalmation']
        data_infer = self.list_word_to_vector(data_infer)
        self.infer(classify, [data_infer])

        data_infer = ['stupid', 'garbage']
        data_infer = self.list_word_to_vector(data_infer)
        self.infer(classify, [data_infer])


if __name__ == '__main__':
    """
    主函数
    """

    data = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
           ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
           ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    label = ["zhao", 'wen', 'feng', 'wen', 'feng', 'wen']

    navie_bayes = NaiveBayes(data, label)
    navie_bayes.main()