%%
clear;clc
data_path = '/media/sdnu/Kzhao/M_center/Fea/Feature/data/Fea_corr/';
cd(data_path)
%% load data 
load('SI_AD');load('SI_NC');load('SI_MC');
SI = [SI_AD;SI_MC];
l1 = zeros(1,length(SI_AD(:,1)))+3;l2 = zeros(1,length(SI_MC(:,1)))+1;
SI_label = cat(2,l1,l2);
SI=mapminmax(SI',0,1)';  % load the data and normalize in each center
% GE
load('GE_AD');load('GE_NC');load('GE_MC');
GE = [GE_AD;GE_MC];
l1 = zeros(1,length(GE_AD(:,1)))+3;l2 = zeros(1,length(GE_MC(:,1)))+1;
GE_label = cat(2,l1,l2);
GE=mapminmax(GE',0,1)';
% HH
load('HH_AD');load('HH_NC');load('HH_MC');
HH = [HH_AD;HH_MC];
l1 = zeros(1,length(HH_AD(:,1)))+3;l2 = zeros(1,length(HH_MC(:,1)))+1;
HH_label = cat(2,l1,l2);
HH=mapminmax(HH',0,1)';
%QL
load('QL_AD');load('QL_NC');load('QL_MC');
QL = [QL_AD;QL_MC];
l1 = zeros(1,length(QL_AD(:,1)))+3;l2 = zeros(1,length(QL_MC(:,1)))+1;
QL_label = cat(2,l1,l2);
QL=mapminmax(QL',0,1)';
%XWH
load('XWH_AD');load('XWH_NC');load('XWH_MC');
XWH = [XWH_AD;XWH_MC];
l1 = zeros(1,length(XWH_AD(:,1)))+3;l2 = zeros(1,length(XWH_MC(:,1)))+1;
XWH_label = cat(2,l1,l2);
XWH=mapminmax(XWH',0,1)';
%XWZ
load('XWZ_AD');load('XWZ_NC');load('XWZ_MC');
XWZ = [XWZ_AD;XWZ_MC];
l1 = zeros(1,length(XWZ_AD(:,1)))+3;l2 = zeros(1,length(XWZ_MC(:,1)))+1;
XWZ_label = cat(2,l1,l2);
XWZ=mapminmax(XWZ',0,1)';
%%
label_all = [SI_label,GE_label,HH_label,QL_label,XWH_label,XWZ_label];
for fea_number =50:250  
  for center_name = 1:6
    switch center_name 
      case 1 
         tr_label = cat(1,GE_label',HH_label',QL_label',XWH_label',XWZ_label');
         tr_data = cat(1,GE,HH,QL,XWH,XWZ);
         test_data = SI;
         test_label = SI_label;
      case 2
         tr_label = cat(1,SI_label',HH_label',QL_label',XWH_label',XWZ_label');
         tr_data = cat(1,SI,HH,QL,XWH,XWZ);   
         test_data = GE;
         test_label = GE_label;
      case 3 
         tr_label = cat(1,SI_label',GE_label',QL_label',XWH_label',XWZ_label');
         tr_data = cat(1,SI,GE,QL,XWH,XWZ);   
         test_data = HH;
         test_label = HH_label;
      case 4
          tr_label = cat(1,SI_label',GE_label',HH_label',XWH_label',XWZ_label');
         tr_data = cat(1,SI,GE,HH,XWH,XWZ);   
         test_data = QL;
         test_label = QL_label;
        case 5
          tr_label = cat(1,SI_label',GE_label',HH_label',QL_label',XWZ_label');
         tr_data = cat(1,SI,GE,HH,QL,XWZ);   
         test_data = XWH;
         test_label = XWH_label;
        case 6
         tr_label = cat(1,SI_label',GE_label',HH_label',QL_label',XWH_label');
         tr_data = cat(1,SI,GE,HH,QL,XWH);   
         test_data = XWZ;
         test_label = XWZ_label;
    end  
           [ftRank,ftScore] = ftSel_SVMRFECBR(tr_data,tr_label,2,1/fea_number);  % feature ranking with SVM-RFE
           train_data2 = tr_data(:,ftRank(1:fea_number));
           test_data2 = test_data(:,ftRank(1:fea_number));
%            [bestacc,bestc,bestg]=SVM_cgchoose(train_data2,tr_label,5);
%            % Also you can select c and g with grid search
           cmd = ['   -c  ', num2str(2) , ' -g ', num2str(1/fea_number),' -q  -b  1'];
           model = svmtrain(tr_label,train_data2,cmd);
           [predict_label,accuracy, decision_value] = svmpredict (test_label',test_data2,model,' -b  1'); 
           acc(fea_number) = accuracy(1);
  end 
 end
disp(num2str(max(acc)))
 



  
  