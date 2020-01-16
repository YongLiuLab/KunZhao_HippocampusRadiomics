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

#初始运行成功版本
'''
data_1_path ="pet/adnc.mat"
data = scio.loadmat(data_1_path)['d']
#print(train_data)
target_path="pet/target_adnc.mat"
target=scio.loadmat(target_path)['target']
'''
'''
data_1_path ="pet/test/smcpmc/data.mat"
data = scio.loadmat(data_1_path)['data']
#print(train_data)
target_path="pet/label1.mat"
target=scio.loadmat(target_path)['label']

#划分训练集和测试集 交叉验证：
X_train,X_test, y_train, y_test =train_test_split(data,target,test_size=0.16, random_state=0)
train_data=X_train
train_target=y_train
test_data=X_test
test_target=y_test
train_target = np.ravel(train_target) #将多维数组转换为一维数组的功能
test_target = np.ravel(test_target)  #将多维数组转换为一维数组的功能
'''
#导入数据
data_1_path = r"C:\Users\Kunzh\Desktop\20200112\Classification\IN_ADNI\data_all.mat"
target_1_path = r"C:\Users\Kunzh\Desktop\20200112\Classification\IN_ADNI\label.mat"

train_data = scio.loadmat(data_1_path)['data_all']
train_target = scio.loadmat(target_1_path)['label']

data_2_path = r"C:\Users\Kunzh\Desktop\20200112\Classification\IN_ADNI\data_adni.mat"
target_2_path = r"C:\Users\Kunzh\Desktop\20200112\Classification\IN_ADNI\label_adni.mat"

test_data=scio.loadmat(data_2_path)['data_adni']
test_target = scio.loadmat(target_2_path)['label_adni']


train_target = np.ravel(train_target) #将多维数组转换为一维数组的功能
test_target = np.ravel(test_target)  #将多维数组转换为一维数组的功能

# print (len(test_data))
# print(len(train_data))
# print (test_data)
# print(test_target)
# print(type(test_data))
# print(type(test_target))
#将数据转换成lightgbm格式
lgb_train = lgb.Dataset(train_data,train_target)
lgb_test=lgb.Dataset(test_data,test_target)
#print(train_data.get_field())
#print(train_data.construct())
#print(train_data.get_group())

params = {
    'objective':'binary',
   'verbose':-1
  }
print("jiaocha...")
cv_results = lgb.cv(params,lgb_train,metrics='binary_error',nfold=10)
print("xunlian")
print(cv_results['binary_error-mean'])
print(pd.Series(cv_results['binary_error-mean']))
mean_merror = pd.Series(cv_results['binary_error-mean']).min()
num_round = pd.Series(cv_results['binary_error-mean']).argmin()+1
print(num_round)
print(mean_merror)


#训练测试

params = {
    'objective':'binary',
    'verbose':-1
    }

gbm = lgb.train(params,lgb_train,num_boost_round=num_round)
print(gbm.num_trees())

#训练集精度


#测试集精度
y_predict = gbm.predict(test_data)
#for i in range(len(y_predict)):
#    if(y_predict[i]<0.5):
 #       y_predict[i]=0
  #  else:
   #     y_predict[i]=1
#print(accuracy_score(test_target,y_predict))
fpr, tpr, thresholds = metrics.roc_curve(test_target, y_predict)
'''
roc_auc=auc(fpr,tpr)
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xlabel('False Positive Rate') #横坐标是fpr
plt.ylabel('True Positive Rate')  #纵坐标是tpr
plt.title('Receiver operating characteristic example')
plt.show()
plt.savefig('xboost_roc.png')
'''
roc_auc=auc(fpr,tpr)
'''
lw=2
plt.figure()
plt.plot(fpr, tpr,
         label=' (AUC = {0:0.3f})'
               '' .format(roc_auc),
         color='red',  linewidth=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.1, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig('xgboost_roc.png')
'''
#recall=recall_score(test_target, y_predict,average='binary')
# print(test_target)
# print(y_predict)
#print("recall")
#print (recall)
# print("fpr")
# print(fpr)
# print("tpr")
# print(tpr)
#print(thresholds)
print("AUC")
print(metrics.auc(fpr, tpr))
print(roc_auc_score(test_target, y_predict))
TP, TN, FP, FN = 0, 0, 0, 0
for index in range(len(y_predict)):
    if (test_target [index] == 1):
        if (y_predict[index]>0.5):
            TN +=  1
        else:
            FP += 1
    else:
        if (y_predict[index]>0.5):
            FN = FN + 1;
        else:
            TP = TP + 1;

# print("TP")
# print(TP)
# print("FN")
# print(FN)
# print("FP")
# print(FP)
# print("TN")
# print(TN)
SEN = TP/(TP + FN)
SPE = TN/(TN + FP)
ACC = (TP + TN)/(TP + FP + FN + TN)
print("SEN")
print(SEN)
print("SPE")
print(SPE)
print("ACC")
print(ACC)
