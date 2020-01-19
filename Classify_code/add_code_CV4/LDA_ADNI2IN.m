   clear;clc
% %PCA降维，降到492*491，按列对主成分进行排序
load('D:\previous_work_brainnetome\MCAD\reviewer_add\LDA\ADNI_IN\label_adni');
load('D:\previous_work_brainnetome\MCAD\reviewer_add\LDA\ADNI_IN\label');
load('D:\previous_work_brainnetome\MCAD\reviewer_add\LDA\ADNI_IN\data_all');
load('D:\previous_work_brainnetome\MCAD\reviewer_add\LDA\ADNI_IN\data_adni');

%[data1,PS]=mapminmax(data,0,1)
options = [ ];
options.PCARatio = 1 ;
[eigvector, eigvalue, meanData, data_adni] = PCA(data_adni, options);
data_all = data_all*eigvector;
data = [data_all;data_adni];
label_inad = [label;label_adni'];

%循环跑pca的前50维，看哪些维数效果最好
test_data = data(1:length(label),:);
train_data = data(length(label)+1:end,:);
train_label = label_adni;
test_label = label;
for i =13
    [model,k,ClassLabel]=LDATraining(train_data(:,1:i),train_label);
    [tmp,target]=LDATesting(test_data(:,1:i),k,model,ClassLabel);
    [TP,TN,FN,FP] = conclusion(target,test_label);
    SEN(i) = TP / (TP + FN);
    SPE(i) = TN / (TN + FP);
    ACC(i) = (TP + TN) / (TP + FP + FN + TN);
    [auc(i), curve] = rocplot(tmp(:,1)-tmp(:,2), test_label, 1, 0);
end

    
