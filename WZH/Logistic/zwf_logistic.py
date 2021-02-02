import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Logistic(object):
    """
    公式矢量化
    """
    def __init__(self):
        """
        init
        """
        pass
    def get_data(self):
        """
        load data
        """
        data = []
        label = []
        with open('./testSet.txt', 'r') as f:
            for line in f:
                str_line = line.strip().split()
                data.append([float(str_line[0]), float(str_line[1]), 1.0])
                label.append([int(str_line[2])])
        return data, label

    def plot_data(self, weights):
        """
        plot data
        """
        data, label = self.get_data()
        data_num = len(data)
        feature_num = len(data[0])
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for index in range(data_num):
            if label[index][0] == 1:
                x1.append(data[index][0])
                y1.append(data[index][1])
            else:
                x2.append(data[index][0])
                y2.append(data[index][1])
        fig = plt.figure()
        ax = fig.add_subplot(111)  # 添加subplot
        ax.scatter(x1, y1, s=20, c='green', marker='s', alpha=.5)  # 绘制正样本
        ax.scatter(x2, y2, s=20, c='red', alpha=.5)  # 绘制负样本

        x3 = np.arange(-3, +3.0, 0.1)
        y3 = (-weights[0] * x3 - weights[2]) / weights[1]
        ax.plot(x3, y3)

        plt.title('DataSet')  # 绘制title
        plt.xlabel('x')
        plt.ylabel('y')  # 绘制label
        plt.show()

    def sigmoid(self, x):
        """
        sigmoid
        """
        return 1.0 / (np.exp(-x) + 1)

    def train_stoc_grad_ascent(self):
        """
        train
        """
        data, label = self.get_data()
        data = np.array(data)
        label = np.array(label)
        data_num, feature_num = data.shape
        iter_num = 500
        alpha = 0.001
        weights = np.ones((feature_num, 1))
        data_trans = np.transpose(data)
        # 矩阵相乘，如果是矩阵 就.dot 如果不是转成 np.mat() *即可
        for itr in range(iter_num):
            list_index_data = list(range(data_num))
            for i in range(data_num):
                alpha = 4 / (itr + i + 1.0) + 0.01
                # 随着迭代次数不断减少，但永远不会减少到0， 因为这里保存了一个常数项。
                index_select = random.choice(list_index_data)
                list_index_data.remove(index_select)

                data_select = np.array([data[index_select]])
                data_select_trans = np.transpose(data_select)
                label_select = np.array(label[[index_select]])

                weights += alpha * data_select_trans.dot(label_select - self.sigmoid(data_select.dot(weights)))


            # h = self.sigmoid(data.dot(weights))
            # error = label - h
            # ww = data_trans.dot(error) * alpha
            # weights += ww
            # weights += alpha * data_trans.dot(label - self.sigmoid(data.dot(weights)))

        print (weights)
        return weights

    def train(self):
        """
        train
        """
        data, label = self.get_data()
        data = np.array(data)
        label = np.array(label)
        data_num = len(data)
        feature_num = len(data[0])
        iter_num = 500
        alpha = 0.001
        weights = np.ones((feature_num, 1))
        data_trans = np.transpose(data)
        # 矩阵相乘，如果是矩阵 就.dot 如果不是转成 np.mat() *即可
        for itr in range(iter_num):
            # h = self.sigmoid(data.dot(weights))
            # error = label - h
            # ww = data_trans.dot(error) * alpha
            # weights += ww
            weights += alpha * data_trans.dot(label - self.sigmoid(data.dot(weights)))
        print (weights)
        return weights

    def main(self):
        """
        main
        """
        # data, label = self.get_data()
        import time
        s = time.time()
        weights = self.train()
        print ('train ', time.time() - s)
        s = time.time()
        weights = self.train_stoc_grad_ascent()
        print('train_2 ', time.time() - s)
        # self.plot_data(weights)




if __name__ == "__main__":
    """
    main function
    """
    data = np.array([[1], [2]])
    label = np.array([[1], [2.5]])
    logistic = Logistic()
    logistic.main()
