import numpy as np

class LinearRegression(object):
    """
    naive bayes
    """
    def __init__(self, data, label):
        """
        init
        """
        # data 是二维数组
        self.data = data
        self.label = label

    def train(self):
        """
        train
        """
        x = np.concatenate((self.data, self.label), axis=1)
        x_trans = np.transpose(x)
        x_trans_multi_x = np.multiply(x_trans, x)
        x_trans_multi_x_inv = np.linalg.inv(x_trans_multi_x)
        result = x_trans_multi_x_inv.dot(x_trans).dot(self.label)
        print (result)
        return result

    def main(self):
        """
        main
        """
        self.train()

if __name__ == "__main__":
    """
    main function
    """
    data = np.array([[1], [2]])
    label = np.array([[1], [2.5]])
    linear_regression = LinearRegression(data, label)
    linear_regression.main()
