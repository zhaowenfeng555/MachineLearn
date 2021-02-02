"""
@ date: 20201028
@ desc: KMeans
"""

import math
import random

class Cluster(object):
    """
    Class Cluster
    """
    def __init__(self):
        """
        init
        """
        self.center = None
        self.element = []

        # element (1, [0.696, 0.774])
        # center 没有编号，是因为可能不是某个确定的输入数据

    def _set_center(self, center):
        """
        set center
        """
        self.center = center

    def _get_center(self):
        """
        get center
        """
        return self.center

    def add_element(self, element):
        """
        add element
        """
        self.element.append(element)

    def update_center(self):
        """
        update center
        """
        length = len(self.element)
        if length == 0:
            return None
        feature_num = len(self.element[0][1])
        for i in range(feature_num):
            current_sum = 0
            for j in range(length):
                current_sum += self.element[j][1][i]
            self.center[i] = round(current_sum / length, 2)

    def distance_center_to_x(self, x):
        """
        distance center to x
        # 这里的 x 是 element_value， 不带具体的编号。
        """
        square = 0
        for i in range(len(x)):
            square += (x[i] - self.center[i]) ** 2
        return math.sqrt(square)

class KMeans(object):
    """
    class KMeans
    """
    def __init__(self, K, ITER_MAX, data):
        """
        init
        """
        self.k = K
        self.iter_max = ITER_MAX
        self.data = data

    def _init_cluster(self):
        """
        init cluster
        """
        self.cluster = [Cluster() for i in range(self.k)]
        self.len_data = len(self.data)

        # 随机选择 self.k 个向量作为中心
        k_index = []
        while len(k_index) < self.k:
            index = random.randint(0, self.len_data - 1)
            if index not in k_index:
                k_index.append(index)
        for i in range(self.k):
            self.cluster[i]._set_center(self.data[k_index[i]])

    def train(self):
        """
        train
        """
        self._init_cluster()

        for iter in range(self.iter_max):
            for i in range(self.k):
                self.cluster[i].element = []

            # 根据每个元素计算到每个簇的距离，选择最小距离，加入该簇
            for j in range(self.len_data):
                element_value = self.data[j]
                element = (j, element_value)

                # 先计算第一个簇到该元素的距离
                min_dist = self.cluster[0].distance_center_to_x(element_value)
                cluster_select = 0

                for k in range(1, self.k):
                    if self.cluster[k].distance_center_to_x(element_value) < min_dist:
                        min_dist = self.cluster[k].distance_center_to_x(element_value)
                        cluster_select = k
                self.cluster[cluster_select].add_element(element)

            # 重新计算均值向量
            flag_update = False
            for i in range(self.k):
                curren_center = self.cluster[i]._get_center()
                self.cluster[i].update_center()
                if curren_center != self.cluster[i]._get_center():
                    flag_update = True

            if flag_update is False:
                break

        # 根据当前的中心点，更新每个簇的数据
        # 根据每个元素计算到每个簇的距离，选择最小距离，加入该簇
        for i in range(self.k):
            self.cluster[i].element = []

        for j in range(self.len_data):
            element_value = self.data[j]
            element = (j, element_value)

            # 先计算第一个簇到该元素的距离
            min_dist = self.cluster[0].distance_center_to_x(element_value)
            cluster_select = 0

            for k in range(1, self.k):
                if self.cluster[k].distance_center_to_x(element_value) < min_dist:
                    min_dist = self.cluster[k].distance_center_to_x(element_value)
                    cluster_select = k
            self.cluster[cluster_select].add_element(element)

    def print(self):
        """
        print cluster
        """
        for i in range(self.k):
            print ('\n\nself.center is ' + str(self.cluster[i]._get_center()))
            for e in self.cluster[i].element:
                print (e)

if __name__ == "__main__":
    """
    main function
    """
    K = 4
    ITER_MAX = 5
    data = [[0.697, 0.460],
            [0.774, 0.376],
            [0.634, 0.264],
            [0.608, 0.318],
            [0.556, 0.215],
            [0.403, 0.237],
            [0.481, 0.149],
            [0.437, 0.211],
            [0.666, 0.091],
            [0.243, 0.267],
            [0.245, 0.057],
            [0.343, 0.057]
            ]
    kmeans = KMeans(K, ITER_MAX, data)
    kmeans.train()
    kmeans.print()





