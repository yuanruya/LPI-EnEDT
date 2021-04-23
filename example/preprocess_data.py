'''
@author:Ruya Yuan
@license: (C) Copyright 2020-2023.
@contact: yuanruyahut@163.com
@time: 2020/10/12
@LPI-EnEDT: An Ensemble Framework with Extra Tree and Decision Tree for Imbalanced LPI Data Classification
'''
import numpy as np
from joblib.numpy_pickle_utils import xrange
from pandas import DataFrame
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
def norm_rna(filename):
    print('normlization')
    RNA_feature = pd.read_csv(filename, header=None, index_col=None).to_numpy()
    data1 = RNA_feature - RNA_feature.mean(axis=0)
    data2 = np.max(RNA_feature,axis=0) - np.min(RNA_feature,axis=0)
    data = data1 / data2
    rna_N = pd.DataFrame(data)
    rna_N.to_csv('D:/learning/lncRNA-protein-t/1/RNA_feature_n.csv', header=None, index=None)

def norm_pro(filename):
    print('normlization')
    PRO_feature = pd.read_csv(filename, header=None, index_col=None).to_numpy()
    data1 = PRO_feature - PRO_feature.mean(axis=0)
    data2 = np.max(PRO_feature,axis=0) - np.min(PRO_feature,axis=0)
    data = data1 / data2
    pro_N = pd.DataFrame(data)
    pro_N = pro_N.dropna(axis=1, how='all')
    pro_N.to_csv('D:/learning/lncRNA-protein-t/1/PRO_feature_n.csv', header=None, index=None)

def dim_rna(filename):
    RNA_feature = pd.read_csv(filename, header=None).to_numpy()
    X_scaled = RNA_feature
    pca = PCA(n_components=100)
    X_scaled = pca.fit_transform(X_scaled)
    scaler = StandardScaler()
    scaler.fit(X_scaled)  # 使用transfrom必须要用fit语句
    trans_data_2 = scaler.transform(X_scaled)  # transfrom通过找中心和缩放等实现标准化
    # trans_data_2 = scaler.fit_transform(sss)  # fit_transfrom为先拟合数据,然后转化它将其转化为标准形式
    # print(trans_data_2.shape)
    ss = DataFrame(trans_data_2)
    ss = ss.dropna(axis=1, how='all')
    # print(trans_data_2.shape)
    ss.to_csv("D:/learning/lncRNA-protein-t/1/RNA_feature_d.csv", index=False, header=None, sep=',')

def dim_pro(filename):
    PRO_feature = pd.read_csv(filename, header=None).to_numpy()
    row,col = PRO_feature.shape
    # print(PRO_feature.shape)
    PRO_feature1 = np.vstack([PRO_feature, PRO_feature])
    # print(PRO_feature1.shape)
    PRO_feature2 = np.vstack([PRO_feature1, PRO_feature])
    # print(PRO_feature2.shape)
    PRO_feature3 = np.vstack([PRO_feature2, PRO_feature])
    PRO_feature4 = np.vstack([PRO_feature3, PRO_feature])
    X_scaled = np.vstack([PRO_feature4, PRO_feature])
    print(X_scaled.shape)
    pca = PCA(n_components=100)
    X_scaled = pca.fit_transform(X_scaled)
    scaler = StandardScaler()
    scaler.fit(X_scaled)  # 使用transfrom必须要用fit语句
    trans_data_2 = scaler.transform(X_scaled)  # transfrom通过找中心和缩放等实现标准化
    # trans_data_2 = scaler.fit_transform(sss)  # fit_transfrom为先拟合数据,然后转化它将其转化为标准形式
    # print(trans_data_2.shape)
    trans_data_2 = trans_data_2[:row]
    # print(trans_data_2.shape)
    ss = DataFrame(trans_data_2)
    ss.to_csv("D:/learning/lncRNA-protein-t/1/PRO_feature_d.csv", index=False, header=None, sep=',')

def contact(rnaf,prof):
    rna = pd.read_csv(rnaf, header=None, index_col=None).to_numpy()
    pro = pd.read_csv(prof, header=None, index_col=None).to_numpy()
    fdim = rna.shape[1] + pro.shape[1]
    feat = np.zeros((1, fdim))
    for i in rna:
        temp = np.array([])
        for j in pro:
            temp = np.hstack([i, j])
            feat = np.vstack([feat, temp])
    feat = feat[1:]
    feat = pd.DataFrame(feat)
    feat.to_csv('D:/learning/lncRNA-protein-t/1/con_feature.csv', header=None, index=None)

def connect(featf, labelf):
    feat = pd.read_csv(featf, header=None, index_col=None).to_numpy()
    label = pd.read_csv(labelf, header=None, index_col=None).to_numpy()
    label = label.flatten().reshape((-1, 1))
    print(label.sum)
    data = pd.DataFrame(np.hstack([feat, label]))
    data.to_csv('D:/learning/lncRNA-protein-t/1/data_con.csv', header=None, index=None)

if __name__ == '__main__':
    # extract_label('land.csv')
    print('Run')
    rna = 'D:/learning/lncRNA-protein-t/1/RNA_f.csv'
    pro = 'D:/learning/lncRNA-protein-t/1/pro_f.csv'
    norm_rna(rna)
    norm_pro(pro)
    dim_rna('D:/learning/lncRNA-protein-t/1/RNA_feature_n.csv')
    dim_pro('D:/learning/lncRNA-protein-t/1/PRO_feature_n.csv')
    contact('D:/learning/lncRNA-protein-t/1/RNA_feature_d.csv', 'D:/learning/lncRNA-protein-t/1/PRO_feature_d.csv')
    connect('D:/learning/lncRNA-protein-t/1/con_feature.csv', 'D:/learning/lncRNA-protein-t/1/label_1.csv')
