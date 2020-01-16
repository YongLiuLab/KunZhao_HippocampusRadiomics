import scipy.io as scio
import numpy as np
import xgboost as xgb
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score,recall_score
import matplotlib.pyplot as plt
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
roc_auc = np.zeros(1000)
ACC =np.zeros(1000)
SEN = np.zeros(1000)
SPE = np.zeros(1000)
for i in range(1000):
    print(i)
    for j in range(6):
       fl = j+1
       center_data = data[center_id==fl,:]
       center_label = target[center_id==fl]
       flag = np.zeros(len(center_label))
       for k in range(len(center_label)):
           flag[k] = k
       rs = random.sample(range(1, len(center_label)), 10)
       train_indx = np.int16([x for x in flag if x not in rs])
       test_data_center = center_data[rs,:]
       test_label_center = center_label[rs]
       train_data_center = center_data[train_indx,:]
       train_label_center = center_label[train_indx]
       if j ==0:
           train_data = train_data_center
           test_data = test_data_center
           train_label = train_label_center
           test_label = test_label_center
       else:

           train_data = np.vstack((train_data,train_data_center))
           test_data = np.vstack((test_data,test_data_center))
           train_label = np.hstack((train_label, train_label_center))
           test_label = np.hstack((test_label, test_label_center))

    xgb_train = xgb.DMatrix(data=train_data, label=train_label)
    xgb_test = xgb.DMatrix(data=test_data, label=test_label)



    #交叉验证
    params = {
        'objective': 'binary:logistic',
        'verbose':-1
        }
    print("jiaocha...")
    cv_results = xgb.cv(params,xgb_train,nfold=10)
    print("xunlian")
    #print(cv_results['binary_error-mean'])
    #print(pd.Series(cv_results['binary_error-mean']))
    #mean_merror = pd.Series(cv_results['binary:logistic']).min()
    #num_round = pd.Series(cv_results['binary:logistic']).argmin()+1
    num_round=200
    print(num_round)
    #print(mean_merror)


    #训练测试
    params_0 = {
        'objective':'binary',
        'verbose':-1
        }

    gbm = xgb.train(params,xgb_train,num_boost_round=num_round)
    #print(gbm.num_trees())
    print ("jixu")
    #训练集精度
    #测试集精度
    y_predict = gbm.predict(xgb_test)

    print(y_predict.shape)
    print("ACC")
    #print(accuracy_score(test_target,y_predict))
    fpr, tpr, thresholds = metrics.roc_curve(test_label, y_predict)
    roc_auc[i]=roc_auc_score(test_label, y_predict)
    TP, TN, FP, FN = 0, 0, 0, 0
    for index in range(len(y_predict)):
        if (test_label [index] == 1):
            if (y_predict[index]>0.5):
                TN +=  1
            else:
                FP += 1
        else:
            if (y_predict[index]>0.5):
                FN = FN + 1;
            else:
                TP = TP + 1;


    SEN[i] = TP/(TP + FN)
    SPE[i] = TN/(TN + FP)
    ACC[i] = (TP + TN)/(TP + FP + FN + TN)
filename = r'E:\previous_work_brainnetome\MCAD\reviewer_add\Classification\intra_site\auc.mat'
scio.savemat(filename, {'roc_auc': roc_auc})
filename = r'E:\previous_work_brainnetome\MCAD\reviewer_add\Classification\intra_site\ACC.mat'
scio.savemat(filename, {'ACC': ACC})
filename = r'E:\previous_work_brainnetome\MCAD\reviewer_add\Classification\intra_site\SEN.mat'
scio.savemat(filename, {'SEN': SEN})
filename = r'E:\previous_work_brainnetome\MCAD\reviewer_add\Classification\intra_site\SPE.mat'
scio.savemat(filename, {'SPE': SPE})