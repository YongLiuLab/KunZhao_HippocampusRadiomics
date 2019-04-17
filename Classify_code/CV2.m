clear;clc
data_path = '';
cd(data_path)
%%
%load
load('SI_AD');load('SI_NC');load('SI_MC');
SI = [SI_AD;SI_NC];
l1 = zeros(1,length(SI_AD(:,1)))+1;l2 = zeros(1,length(SI_NC(:,1)))+2;
SI_label = cat(2,l1,l2);
SI_data=mapminmax(SI',0,1)';
% GE
load('GE_AD');load('GE_NC');load('GE_MC');
GE = [GE_AD;GE_NC];
l1 = zeros(1,length(GE_AD(:,1)))+1;l2 = zeros(1,length(GE_NC(:,1)))+2;
GE_label = cat(2,l1,l2);
GE_data=mapminmax(GE',0,1)';
% HH
load('HH_AD');load('HH_NC');load('HH_MC');
HH = [HH_AD;HH_NC];
l1 = zeros(1,length(HH_AD(:,1)))+1;l2 = zeros(1,length(HH_NC(:,1)))+2;
HH_label = cat(2,l1,l2);
HH_data=mapminmax(HH',0,1)';
%QL
load('QL_AD');load('QL_NC');load('QL_MC');
QL = [QL_AD;QL_NC];
l1 = zeros(1,length(QL_AD(:,1)))+1;l2 = zeros(1,length(QL_NC(:,1)))+2;
QL_label = cat(2,l1,l2);
QL_data=mapminmax(QL',0,1)';
%XWH
load('XWH_AD');load('XWH_NC');load('XWH_MC');
XWH = [XWH_AD;XWH_NC];
l1 = zeros(1,length(XWH_AD(:,1)))+1;l2 = zeros(1,length(XWH_NC(:,1)))+2;
XWH_label = cat(2,l1,l2);
XWH_data=mapminmax(XWH',0,1)';
%XWZ
load('XWZ_AD');load('XWZ_NC');load('XWZ_MC');
XWZ = [XWZ_AD;XWZ_NC];
l1 = zeros(1,length(XWZ_AD(:,1)))+1;l2 = zeros(1,length(XWZ_NC(:,1)))+2;
XWZ_label = cat(2,l1,l2);
XWZ_data=mapminmax(XWZ',0,1)';
for fea_number = 57   %  choose a fea_number, in this paper, the fea_number=57
        for epoh = 1:1000
            SI_test_indx = randperm(length(SI_label),10);
            GE_test_indx = randperm(length(GE_label),10);
            HH_test_indx =  randperm(length(HH_label),10);
            QL_test_indx = randperm(length(QL_label),10);
            XWH_test_indx = randperm(length(XWH_label),10);
            XWZ_test_indx = randperm(length(XWZ_label),10);
            SI_train_indx = setdiff(1:length(SI_label),SI_test_indx);
            GE_train_indx = setdiff(1:length(GE_label),GE_test_indx);
            HH_train_indx = setdiff(1:length(HH_label),HH_test_indx);
            QL_train_indx = setdiff(1:length(QL_label),QL_test_indx);
            XWH_train_indx = setdiff(1:length(XWH_label),XWH_test_indx);
            XWZ_train_indx = setdiff(1:length(XWZ_label),XWZ_test_indx);
            %%
            SI_trainlabel =SI_label(SI_train_indx);
            GE_trainlabel = GE_label(GE_train_indx);
            HH_trainlabel = HH_label(HH_train_indx);
            QL_trainlabel = QL_label(QL_train_indx);
            XWH_trainlabel = XWH_label(XWH_train_indx);
            XWZ_trainlabel = XWZ_label(XWZ_train_indx);
           %%
            SI_testlabel = SI_label(SI_test_indx);
            GE_testlabel = GE_label(GE_test_indx);
            HH_testlabel = HH_label(HH_test_indx);
            QL_testlabel = QL_label(QL_test_indx);
            XWH_testlabel = XWH_label(XWH_test_indx);
            XWZ_testlabel = XWZ_label(XWZ_test_indx);
            %%
            SI_testdata = SI_data(SI_test_indx,:);
            GE_testdata = GE_data(GE_test_indx,:);
            HH_testdata = HH_data(HH_test_indx,:);
            QL_testdata = QL_data(QL_test_indx,:);
            XWH_testdata = XWH_data(XWH_test_indx,:);
            XWZ_testdata = XWZ_data(XWZ_test_indx,:);
            %%
            test_data = cat(1,SI_testdata,GE_testdata,HH_testdata,QL_testdata,XWH_testdata,XWZ_testdata);
            test_label = cat(1,SI_testlabel',GE_testlabel',HH_testlabel',QL_testlabel',XWH_testlabel',XWZ_testlabel');
            %%
            SI_traindata = SI_data(SI_train_indx,:);   
            GE_traindata = GE_data(GE_train_indx,:);
            HH_traindata = HH_data(HH_train_indx,:);
            QL_traindata = QL_data(QL_train_indx,:);
            XWH_traindata = XWH_data(XWH_train_indx,:);
            XWZ_traindata = XWZ_data(XWZ_train_indx,:);
            %%
            train_data = cat(1,SI_traindata,GE_traindata,HH_traindata,QL_traindata,XWH_traindata,XWZ_traindata);
            train_label = cat(1,SI_trainlabel',GE_trainlabel',HH_trainlabel',QL_trainlabel',XWH_trainlabel',XWZ_trainlabel');
            [ftRank,ftScore] = ftSel_SVMRFECBR(train_data,train_label,2^1,1/fea_number);
            train_data = train_data(:,ftRank(1:fea_number));
            test_data = test_data(:,ftRank(1:fea_number));
            cmd = ['-c ', num2str(2) , ' -g ', num2str(1/fea_number), ' -b  1  -q ' ];
            model = svmtrain(train_label,train_data,cmd);
            [predict_label,accuracy, decision_value] = svmpredict (test_label,test_data,model,' -b  1');  
            acc(epoh) = accuracy(1);
        end
            
end
           disp(num2str(max(acc)))
