   clear;clc
% %PCA降维，降到492*491，按列对主成分进行排序
load('C:\Users\Kunzh\Desktop\20200112\Classification\LDA\ADNI_IN\label_adni');
load('C:\Users\Kunzh\Desktop\20200112\Classification\LDA\ADNI_IN\label');
load('C:\Users\Kunzh\Desktop\20200112\Classification\LDA\ADNI_IN\data_all');
load('C:\Users\Kunzh\Desktop\20200112\Classification\LDA\ADNI_IN\data_adni');
data_inad = [data_all;data_adni];
label_inad = [label;label_adni'];
%[data1,PS]=mapminmax(data,0,1)
options = [ ];
options.PCARatio = 1 ;
[eigvector, eigvalue, meanData, data] = PCA(data_inad, options);


test_data = data(1:length(label),:);
train_data = data(length(label)+1:end,:);
train_label= label_adni;
test_label = label;
for i =5
    [model,k,ClassLabel]=LDATraining(train_data(:,1:i),train_label);
    [tmp,target]=LDATesting(test_data(:,1:i),k,model,ClassLabel);
    [TP,TN,FN,FP] = conclusion(target,test_label);
    SEN(i) = TP / (TP + FN);
    SPE(i) = TN / (TN + FP);
    ACC(i) = (TP + TN) / (TP + FP + FN + TN);
    [auc(i), curve] = rocplot(tmp(:,1)-tmp(:,2), test_label, 1, 0);
end

    
