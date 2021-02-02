import numpy as np
from sklearn.linear_model import LogisticRegression

class LR(object):
    """
    naive bayes
    """
    def __init__(self):
        """
        init
        """
        pass

    def get_data(self, test_or_train):
        """
        load data
        """
        if test_or_train == 'train':
            file = './horseColicTraining.txt'
        else:
            file = './horseColicTest.txt'

        data = []
        label = []
        with open(file, 'r') as f:
            for line in f:
                list_line = line.strip().split('\t')
                list_line = list(map(lambda x: float(x), list_line))
                data.append(list_line[: -1])
                label.append(float(list_line[-1]))
        return data, label

    def train(self):
        """
        train
        """
        data_train, label_train = self.get_data('train')
        test_train, label_test = self.get_data('test')
        classifier = LogisticRegression(solver='sag', max_iter=500).fit(data_train, label_train)
        test_accuracy = classifier.score(test_train, label_test) * 100
        print('正确率:%f%%' % test_accuracy)




    def main(self):
        """
        main
        """
        self.train()

if __name__ == "__main__":
    """
    main function
    """
    logistic_regression = LR()
    logistic_regression.main()
