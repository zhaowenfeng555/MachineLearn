import numpy as np
from sklearn.metrics import roc_auc_score


def calc_auc(y_labels, y_scores):
    f = list(zip(y_scores, y_labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]

    print (len(rank))
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    print (len(rankList))
    print (rankList)
    pos_cnt = np.sum(y_labels == 1)
    neg_cnt = np.sum(y_labels == 0)
    auc = (np.sum(rankList) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)
    print(auc)


def calc_auc2(y_labels, y_scores):
    zip_labels_score = list(zip(y_labels, y_scores))
    list_sorted_labels = [v1 for v1, v2 in sorted(zip_labels_score, key = lambda x: x[1])]
    list_pos_number = [i+1 for i in range(len(list_sorted_labels)) if list_sorted_labels[i] == 1]
    m = np.sum(y_labels == 1)
    n = np.sum(y_labels == 0)

    auc = (np.sum(list_pos_number) - m * (m + 1)/2)/(m * n)


    print(auc)


def get_score():
    # 随机生成100组label和score
    y_labels = np.zeros(100)
    y_scores = np.zeros(100)
    for i in range(100):
        y_labels[i] = np.random.choice([0, 1])
        y_scores[i] = np.random.random()
    return y_labels, y_scores


if __name__ == '__main__':
    y_labels, y_scores = get_score()
    # 调用sklearn中的方法计算AUC，与后面自己写的方法作对比
    print('sklearn AUC:', roc_auc_score(y_labels, y_scores))
    calc_auc(y_labels, y_scores)
    calc_auc2(y_labels, y_scores)