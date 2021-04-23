'''
@author:Ruya Yuan
@license: (C) Copyright 2020-2023.
@contact: yuanruyahut@163.com
@time: 2020/10/12
@LPI-EnEDT: An Ensemble Framework with Extra Tree and Decision Tree for Imbalanced LPI Data Classification
'''

import EnEDT
import numpy as np
import pandas as pd
import random
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score

def kfold(data, k, row=0, col=0, cv=3):
    dlen = len(data)
    if cv == 1:
        lens = row
    elif cv == 2:
        lens = col
    else:
        lens = dlen
    d = list(range(lens))
    random.shuffle(d)
    test_n = lens // k
    n = lens % k
    test_res = []
    train_res = []
    for i in range(n):
        test = d[i * (test_n + 1):(i + 1) * (test_n + 1)]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    for i in range(n, k):
        test = d[i * test_n + n:(i + 1) * test_n + n]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    if cv == 3:
        return train_res, test_res
    else:
        train_s = []
        test_s = []
        for i in range(k):
            train_ = []
            test_ = []
            for j in range(dlen):
                if data[j][cv - 1] in test_res[i]:
                    test_.append(j)
                else:
                    train_.append(j)
            train_s.append(train_)
            test_s.append(test_)
        return train_s, test_s
time = 5
k = 5
cv = 3
PREs = np.array([])
ACCs = np.array([])
RECs = np.array([])
AUCs = np.array([])
AUPRs = np.array([])
F1s = np.array([])
data = pd.read_csv('./data1/data.csv', header=None, index_col=None).to_numpy()
label = pd.read_csv('./data1/label.csv', index_col=0).to_numpy()
row, col = label.shape
p = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j]])
n = np.array([(i, j) for i in range(row) for j in range(col) if label[i][j] == 0])
np.random.shuffle(n)
sample = len(p)
n = n[:sample]
d = np.vstack([p,n])
for j in range(time):
    if cv == 3:
        p_tr, p_te = np.array(kfold(p, k=k, row=row, col=col, cv=cv), dtype=object)
        n_tr, n_te = np.array(kfold(n, k=k, row=row, col=col, cv=cv), dtype=object)
    else:
        d_tr, d_te = np.array(kfold(d, k=k, row=row, col=col, cv=cv), dtype=object)
    for i in range(k):
        if cv == 3:
            train_sample = np.vstack([np.array(p[p_tr[i]]), np.array(n[n_tr[i]])])
            test_sample = np.vstack([np.array(p[p_te[i]]), np.array(n[n_te[i]])])
        else:
            train_sample = np.array(d[d_tr[i]])
            test_sample = np.array(d[d_te[i]])
        train_land = train_sample[:, 0] * col + train_sample[:, 1]
        test_land = test_sample[:, 0] * col + test_sample[:, 1]
        np.random.shuffle(train_land)
        np.random.shuffle(test_land)
        X_tr = data[train_land][:, :-1]
        y_tr = data[train_land][:, -1]
        X_te = data[test_land][:, :-1]
        y_te = data[test_land][:, -1]
        label = y_te
        model = EnEDT.AdaBoost(n_estimators=10, depth=5, split=5, neighbours=3)
        model.fit(X_tr, y_tr)
        score = model.predict_proba1(X_te)[:, 1]
        pre_label = np.zeros(score.shape)
        pre_label = model.predict1(X_te)[0]
        acc = accuracy_score(label, pre_label)
        rec = recall_score(label, pre_label)
        f1 = f1_score(label, pre_label)
        pre = precision_score(label, pre_label)
        fp, tp, threshold = roc_curve(label, score)
        pre_, rec_, _ = precision_recall_curve(label, score)
        au = auc(fp, tp)
        aupr = auc(rec_, pre_)
        PREs = np.append(PREs, pre)
        ACCs = np.append(ACCs, acc)
        RECs = np.append(RECs, rec)
        AUCs = np.append(AUCs, au)
        AUPRs = np.append(AUPRs, aupr)
        F1s = np.append(F1s, f1)

        print('In time {}, k = {}:'.format(j + 1, i + 1))
PRE = PREs.mean()
ACC = ACCs.mean()
REC = RECs.mean()
AUC = AUCs.mean()
AUPR = AUPRs.mean()
F1 = F1s.mean()

PRE_err = np.std(PREs)
ACC_err = np.std(ACCs)
REC_err = np.std(RECs)
AUC_err = np.std(AUCs)
AUPR_err = np.std(AUPRs)
F1_err = np.std(F1s)

print('\n')
print("PRE is:{}±{}".format(round(PRE, 4), round(PRE_err, 4)))
print("REC is:{}±{}".format(round(REC, 4), round(REC_err, 4)))
print("ACC is:{}±{}".format(round(ACC, 4), round(ACC_err, 4)))
print("F1 is:{}±{}".format(round(F1, 4), round(F1_err, 4)))
print('AUC is :{}±{}'.format(round(AUC, 4), round(ACC_err, 4)))
print('AUPR is :{}±{}'.format(round(AUPR, 4), round(AUPR_err, 4)))
