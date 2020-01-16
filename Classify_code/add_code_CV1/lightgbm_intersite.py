import scipy.io as scio
import numpy as np
np.set_printoptions(threshold=np.inf)
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import random
#初始运行成功版本

#导入数据
data_1_path = r"E:\previous_work_brainnetome\MCAD\reviewer_add\Classification\intra_site\data_all.mat"
target_1_path = r"E:\previous_work_brainnetome\MCAD\reviewer_add\Classification\intra_site\label.mat"
center_path = r'E:\previous_work_brainnetome\MCAD\reviewer_add\Classification\intra_site\center.mat'

data = scio.loadmat(data_1_path)['data_all']
target = scio.loadmat(target_1_path)['label']
center_id = scio.loadmat(center_path)['center']

target = np.ravel(target)  #将多维数组转换为一维数组的功能
center_id = np.ravel(center_id)
roc_auc, ACC, SEN, SPE = np.zeros(1000)

for j in range(6):
    fl = j+1
    test_data = data[center_id==fl,:]
    test_label = target[center_id==fl]
    train_data = data[center_id != fl, :]
    train_label = target[center_id != fl]



    lgb_train = lgb.Dataset(train_data, train_label)
    lgb_test = lgb.Dataset(test_data, test_label)
    # print(train_data.get_field())
    # print(train_data.construct())
    # print(train_data.get_group())

    params = {
        'objective': 'binary',
        'verbose': -1
    }
    cv_results = lgb.cv(params, lgb_train, metrics='binary_error', nfold=10)
    mean_merror = pd.Series(cv_results['binary_error-mean']).min()
    num_round = pd.Series(cv_results['binary_error-mean']).argmin() + 1

    # 训练测试

    params = {
        'objective': 'binary',
        'verbose': -1
    }

    gbm = lgb.train(params, lgb_train, num_boost_round=num_round)
    print(gbm.num_trees())

    # 训练集精度

    # 测试集精度
    y_predict = gbm.predict(test_data)
    # for i in range(len(y_predict)):
    #    if(y_predict[i]<0.5):
    #       y_predict[i]=0
    #  else:
    #     y_predict[i]=1
    # print(accuracy_score(test_target,y_predict))
    fpr, tpr, thresholds = metrics.roc_curve(test_label, y_predict)

    roc_auc[j]=roc_auc_score(test_label, y_predict)
    print(roc_auc_score(test_label, y_predict))
    TP, TN, FP, FN = 0, 0, 0, 0
    for index in range(len(y_predict)):
        if (test_label[index] == 1):
            if (y_predict[index] > 0.5):
                TN += 1
            else:
                FP += 1
        else:
            if (y_predict[index] > 0.5):
                FN = FN + 1;
            else:
                TP = TP + 1;


    SEN[j] = TP / (TP + FN)
    SPE[j] = TN / (TN + FP)
    ACC[j] = (TP + TN) / (TP + FP + FN + TN)

