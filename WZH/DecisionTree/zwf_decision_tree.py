import numpy as np
import math
import pickle
from collections import Counter




class DecisionTree(object):
    """
    naive bayes
    """
    def __init__(self, data, feature_map):
        """
        init
        """
        # data 是二维数组
        self.data = data
        self.feature_map = feature_map

    def get_information_entropy(self, sub_data):
        """
        获取信息熵
        """
        num_entries = len(sub_data)
        list_label = [entries[-1] for entries in sub_data]
        counter = Counter(list_label)
        sum_entropy = 0
        for k, v in dict(counter).items():
            prob = float(v) / num_entries
            sum_entropy -= prob * math.log(prob, 2)
        return round(sum_entropy, 2)

    def split_data(self, sub_data, axis, value):
        """
        把该轴上的该值切分出来
        """
        ret_sub_data = []
        for entries in sub_data:
            if entries[axis] == value:
                half_entries = entries[: axis]
                half_entries.extend(entries[axis+1:])
                ret_sub_data.append(half_entries)
        return ret_sub_data

    def get_best_feature(self, sub_data):
        """
        get best feature
        """
        if sub_data is None:
            return 0
        base_entropy = self.get_information_entropy(sub_data)
        num_entries = len(sub_data)
        num_feature = len(sub_data[0]) - 1

        information_gain = 0
        best_feature = 0

        for feature in range(num_feature):
            feature_values = [entries[feature] for entries in sub_data]
            entropy_condition = 0.0
            for value in set(feature_values):
                ret_sub_data = self.split_data(sub_data, feature, value)
                prob = float(len(ret_sub_data)) / num_entries
                entropy_condition += prob * self.get_information_entropy(ret_sub_data)

            current_ig = base_entropy - entropy_condition
            if current_ig > information_gain:
                information_gain = current_ig
                best_feature = feature
        return best_feature

    def majority(self, list_label):
        """
        样本最多的类为该类
        """
        counter = Counter(list_label)
        # counter.most_common 返回值为： [('key', 'value')]
        return counter.most_common(1)[0][0]

    def train(self, data, feature_map, feat_):
        """
        train
        """
        list_label = [entries[-1] for entries in data]
        # 属于同类样本
        if len(Counter(list_label)) == 1:
            return list_label[0]
        # 属性集为空， 或者所有样本在所有属性上取值相同
        if len(feature_map) == 0:
            return self.majority(list_label)
        # 当前节点包含的样本集合为空， 不能划分


        best_feature = self.get_best_feature(data)
        best_feature_mapping = feature_map[best_feature]
        feat_.append(best_feature_mapping)

        # 去掉 feature_map 中的best
        del feature_map[best_feature]
        my_tree = {best_feature_mapping: {}}
        feature_values = [entries[best_feature] for entries in data]
        for value in set(feature_values):
            ret_sub_feature_map = feature_map[:]
            ret_sub_data = self.split_data(data, best_feature, value)
            my_tree[best_feature_mapping][value] = self.train(ret_sub_data, ret_sub_feature_map, feat_)

        return my_tree

    def infer(self, my_tree, infer, feature_map_back):
        """
        infer
        """
        dict_infer = dict(zip(feature_map_back, infer))
        print (dict_infer)

        first_key = next(iter(my_tree))
        second_dict = my_tree[first_key]
        for key_concrete in second_dict.keys():
            if dict_infer[first_key] == key_concrete:
                if type(second_dict[key_concrete]).__name__ == 'dict':
                    result = self.infer(second_dict[key_concrete], infer)
                else:
                    result = second_dict[key_concrete]
        print (result)
        return result

    def store_my_tree(self, my_tree, file_store):
        """
        存储
        """
        with open(file_store, 'wb') as f:
            pickle.dump(my_tree, f)

    def anti_store_my_tree(self, file_store):
        """
        反序列化
        """
        with open(file_store, 'rb') as f:
            return pickle.load(f)

    def main(self):
        """
        main
        """
        # entropy = self.get_information_entropy(self.data)
        # print (entropy)
        # print (self.get_best_feature(self.data))
        feature_map_back = self.feature_map[:]
        feat_ = []
        my_tree = self.train(self.data, self.feature_map, feat_)
        print (my_tree)

        # 序列化
        file_store = './file_store_my_tree.txt'
        self.store_my_tree(my_tree, file_store)

        # 反序列化
        my_tree_from_file = self.anti_store_my_tree(file_store)

        infer = [1, 0, 1, 2]
        self.infer(my_tree_from_file, infer, feature_map_back)


if __name__ == "__main__":
    """
    main function
    """
    data = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]

    # data = [[0, 0, 0, 0, 'no'],
    #         [0, 0, 0, 0, 'no'],
    #         [0, 0, 0, 0, 'yes']]
    feature_map = ['年龄', '有工作', '有自己的房子', '信贷情况']
    model = DecisionTree(data, feature_map)
    model.main()
