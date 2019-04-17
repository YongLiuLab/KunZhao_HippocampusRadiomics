clear;clc
data_path = '';
cd(data_path)
%%
%load in-house data
%%
load('SI_AD');load('SI_NC');load('SI_MC');
SI = [SI_NC;SI_MC;SI_AD];  % Change the group 
l1 = zeros(1,length(SI_NC(:,1)))+1;l3 = zeros(1,length(SI_AD(:,1)))+3;l2 = zeros(1,length(SI_MC(:,1)))+2;
SI_label = cat(2,l1,l2,l3);
SI=mapminmax(SI',0,1)';
%%
% GE
load('GE_AD');load('GE_NC');load('GE_MC');
GE = [GE_NC;GE_MC;GE_AD];
l1 = zeros(1,length(GE_NC(:,1)))+1;l3 = zeros(1,length(GE_AD(:,1)))+3;l2 = zeros(1,length(GE_MC(:,1)))+2;
GE_label = cat(2,l1,l2,l3);
GE=mapminmax(GE',0,1)';
%%
% HH
load('HH_AD');load('HH_NC');load('HH_MC');
HH = [HH_NC;HH_MC;HH_AD];
l1 = zeros(1,length(HH_NC(:,1)))+1;l2 = zeros(1,length(HH_MC(:,1)))+2;l3 = zeros(1,length(HH_AD(:,1)))+3;
HH_label = cat(2,l1,l2,l3);
HH=mapminmax(HH',0,1)';
%%
%QL
load('QL_AD');load('QL_NC');load('QL_MC');
QL = [QL_NC;QL_MC;QL_AD];
l1 = zeros(1,length(QL_NC(:,1)))+1;l2 = zeros(1,length(QL_MC(:,1)))+2;l3 = zeros(1,length(QL_AD(:,1)))+3;
QL_label = cat(2,l1,l2,l3);
QL=mapminmax(QL',0,1)';
%%
%XWH
load('XWH_AD');load('XWH_NC');load('XWH_MC');
XWH = [XWH_NC;XWH_MC;XWH_AD];
l1 = zeros(1,length(XWH_NC(:,1)))+1;l2 = zeros(1,length(XWH_MC(:,1)))+2;l3 = zeros(1,length(XWH_AD(:,1)))+3;
XWH_label = cat(2,l1,l2,l3);
XWH=mapminmax(XWH',0,1)';
%%
%XWZ
load('XWZ_AD');load('XWZ_NC');load('XWZ_MC');
XWZ = [XWZ_NC;XWZ_MC;XWZ_AD];
l1 = zeros(1,length(XWZ_NC(:,1)))+1;l2 = zeros(1,length(XWZ_MC(:,1)))+2;l3 = zeros(1,length(XWZ_AD(:,1)))+3;
XWZ_label = cat(2,l1,l2,l3);
XWZ=mapminmax(XWZ',0,1)';

%%
%ADNI
test_data = [SI;GE;HH;QL;XWH;XWZ];
test_data = mapminmax(test_data',0,1)';
test_label = [SI_label';GE_label';HH_label';QL_label';XWH_label';XWZ_label'];
load('M:\M_center\MCAD\ADNI_data\ADNI_new');
ADNI_AD_data = ADNI.data(find(ADNI.group==3),:);
ADNI_NC_data = ADNI.data(find(ADNI.group==1),:);
ADNI_MCI_data = ADNI.data(find(ADNI.group==2),:);
ADNI = [ADNI_AD_data;ADNI_NC_data];
l1 = zeros(1,length(ADNI_AD_data(:,1)))+1;l2 = zeros(1,length(ADNI_NC_data(:,1)))+2;%l3 = zeros(1,length(ADNI_MCI_data(:,1)))+2;
ADNI_label = cat(2,l1,l2);
ADNI=mapminmax(ADNI',0,1)';
%%
train_label = ADNI_label;
train_data = ADNI;
for fea_number =50:250
         [ftRank,ftScore] = ftSel_SVMRFECBR(train_data,train_label',2,1/fea_number);
         train_data2 = train_data(:,ftRank(1:fea_number));
         test_data2 = test_data(:,ftRank(1:fea_number));
         cmd = [' -c   ',num2str(2),'   -g  ', num2str(1/fea_number),'  -q  '];
         model = svmtrain(train_label',train_data2,cmd);
         test_data1 = test_data(:,ftRank(1:fea_number));
         [predict_label,accuracy,decision_value] = svmpredict (test_label,test_data1,model);
         acc(fea_number) = accuracy(1);  
end
disp(num2str(max(acc)))