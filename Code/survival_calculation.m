clear all;clear;clc;

addpath('/home/ccf/matlab_project/libsvm-3.22/matlab/');
[num, txt, raw] = xlsread('./survival_data/2011_calculation_275.xlsx', 'A1:L276');
path = '/home/ccf/CCF/Colorecal-cancer/2011_survival/image_feature/code/features_mat/';
savepath = '/home/ccf/CCF/Colorecal-cancer/2011_survival/random_region_experiment/result/';
% load([path, 'CGTfeats.mat']);
% load([path, 'Clustergraphfeats.mat']);
% load([path, 'Graphfeature.mat']);
% load([path, 'Morphologicalfeature.mat']);
% feature_file = load([path,'features_name_275.mat']);    %55 dim tissues proportion features
% featureALL = feature_file.diff_tissue_proportion;
% CGTfeats = cell2mat(CGTfeats);
% Clustergraphfeats = cell2mat(Clustergraphfeats);  
% Graphfeature = cell2mat(Graphfeature);
% Morphologicalfeature = cell2mat(Morphologicalfeature);
% featureALL = [CGTfeats Clustergraphfeats Graphfeature Morphologicalfeature featureALL];
load([path,'Texturefeats.mat']);                                       % 720 dim texture features
featureALL = Texturefeats(:,1:480);
% featureALL = [Texturefeats(:,1:480) featureALL];
[~, dim] = size(featureALL);
% name = feature_file.name;
feature_nums = dim;
features = [];
for i = 1: feature_nums
    feature = mapminmax(featureALL(:,i)');
    features = [features;feature];
end
features = features';
labels = num(:,9);
labels(labels<60) = 0;
labels(labels>=60) = 1;
censor = num(:,10);
survival_time = num(:,9);
clear feature_file;

[m, ~] = size(features);
random_num = randperm(m);
k = ceil(m/3);
train_label = labels(random_num(k:m),:);
train_data = features(random_num(k:m),:);
test_label = labels(random_num(1:k),:);
test_data = features(random_num(1:k),:);
raw_new = raw(2:end,:); 
clinical_train_data = raw_new(random_num(k:m),:);
clinincal_test_data = raw_new(random_num(1:k),:);


iteration_num = 100;
cross_vaildation = 5;
feature_num =5;
for j = 1 : iteration_num
    [a ,b] = size(train_data);
    random = randperm(a);
    data = train_data(random(ceil(a/5):a),:);
    label = train_label(random(ceil(a/5):a),:);
%     data = makeDataDiscrete_mrmr(data);
    [fea(j,:)] = mrmr_mid_d(data, label, feature_num);
%     [fea(j,:)] = mrmr_miq_d(data, label, feature_num);
%     [fea(j,:)] = Wilkcoxnew(data, label, feature_num);
%     [fea(j,:)] = ttestnew(data, label, feature_num);
end

frequency = tabulate(fea(:));
frequency1 = sortrows(frequency,-2);
train_feature_select = train_data(:,frequency1(1:feature_num,1));
data_test = test_data(:,frequency1(1:feature_num,1));

% %SVM
param = SVM(train_feature_select, train_label, data_test, test_label);
test_survival_time = survival_time(random_num(1:k),:);
test_censor = censor(random_num(1:k),:);
predict_label_test_bestmodel = param.predict_label_test_bestmodel_SVM;
test_survival_time_short = test_survival_time(find(predict_label_test_bestmodel==0));
test_survival_time_long = test_survival_time(find(predict_label_test_bestmodel==1));
test_censor_short = test_censor(find(predict_label_test_bestmodel==0));
test_censor_long = test_censor(find(predict_label_test_bestmodel==1));
logrank_v2([test_survival_time_short,test_censor_short],[test_survival_time_long,test_censor_long], 0.05);
plot(param.FPR_test_SVM,param.TPR_test_SVM,'g','LineWidth',1)
save([savepath,'480dim','/','mrmr','/','ALL_para_SVM_mrmr.mat']);
hold on;



%LDA
param = LDA(train_feature_select, train_label, data_test, test_label);
test_survival_time = survival_time(random_num(1:k),:);
test_censor = censor(random_num(1:k),:);
predict_label_test_bestmodel = param.predict_label_test_bestmodel_LDA;
test_survival_time_short = test_survival_time(find(predict_label_test_bestmodel==0));
test_survival_time_long = test_survival_time(find(predict_label_test_bestmodel==1));
test_censor_short = test_censor(find(predict_label_test_bestmodel==0));
test_censor_long = test_censor(find(predict_label_test_bestmodel==1));
logrank_v2([test_survival_time_short,test_censor_short],[test_survival_time_long,test_censor_long], 0.05);
plot(param.FPR_test_LDA,param.TPR_test_LDA,'b','LineWidth',1)
save([savepath,'480dim','/','mrmr','/','ALL_para_LDA_mrmr.mat']);
% 

% 
% %QDA
param = QDA(train_feature_select, train_label, data_test, test_label);
 
test_survival_time = survival_time(random_num(1:k),:);
test_censor = censor(random_num(1:k),:);
predict_label_test_bestmodel = param.predict_label_test_bestmodel_QDA;
test_survival_time_short = test_survival_time(find(predict_label_test_bestmodel==0));
test_survival_time_long = test_survival_time(find(predict_label_test_bestmodel==1));
test_censor_short = test_censor(find(predict_label_test_bestmodel==0));
test_censor_long = test_censor(find(predict_label_test_bestmodel==1));
logrank_v2([test_survival_time_short,test_censor_short],[test_survival_time_long,test_censor_long], 0.05);
plot(param.FPR_test_QDA,param.TPR_test_QDA,'r','LineWidth',1)
save([savepath,'480dim','/','mrmr','/','ALL_para_QDA_mrmr.mat']);

% 
% 
% %KNN
param = KNN(train_feature_select, train_label, data_test, test_label);

test_survival_time = survival_time(random_num(1:k),:);
test_censor = censor(random_num(1:k),:);
predict_label_test_bestmodel = param.predict_label_test_bestmodel_KNN;
test_survival_time_short = test_survival_time(find(predict_label_test_bestmodel==0));
test_survival_time_long = test_survival_time(find(predict_label_test_bestmodel==1));
test_censor_short = test_censor(find(predict_label_test_bestmodel==0));
test_censor_long = test_censor(find(predict_label_test_bestmodel==1));
logrank_v2([test_survival_time_short,test_censor_short],[test_survival_time_long,test_censor_long], 0.05);
plot(param.FPR_test_KNN,param.TPR_test_KNN,'r','LineWidth',1)
save([savepath,'480dim','/','mrmr','/','ALL_para_KNN_mrmr.mat']);



% %RF
param = RF(train_feature_select, train_label, data_test, test_label);

test_survival_time = survival_time(random_num(1:k),:);
test_censor = censor(random_num(1:k),:);
predict_label_test_bestmodel = str2num(char(param.predict_label_test_bestmodel_RF));
test_survival_time_short = test_survival_time(find(predict_label_test_bestmodel==0));
test_survival_time_long = test_survival_time(find(predict_label_test_bestmodel==1));
test_censor_short = test_censor(find(predict_label_test_bestmodel==0));
test_censor_long = test_censor(find(predict_label_test_bestmodel==1));
logrank_v2([test_survival_time_short,test_censor_short],[test_survival_time_long,test_censor_long], 0.05);
plot(param.FPR_test_RF,param.TPR_test_RF,'r','LineWidth',1)
save([savepath,'480dim','/','mrmr','/','ALL_para_RF_mrmr.mat']);


%clinincal variable
%T,(T0,Tis,T1,T2)vs.(T3,T4)
modelset = clinical_train_data;
testset = clinincal_test_data;
a = 1; b = 1;
for i =1:numel(modelset(:,4))
    if (strcmp(modelset{i,4},'T0') || strcmp(modelset{i,4},'T1') || strcmp(modelset{i,4},'T2') || strcmp(modelset{i,4},'Tis'))
        modelset_T_low_sur_time(a,1) = modelset{i,11};
        modelset_T_low_sur_censor(a,1) = modelset{i,12};
        T_modelset(i,1) = 0;
        a = a+1;
    elseif (strcmp(modelset{i,4},'T3') || strcmp(modelset{i,4},'T4'))
        modelset_T_high_sur_time(b,1) = modelset{i,11};
        modelset_T_high_sur_censor(b,1) = modelset{i,12};
        T_modelset(i,1) = 1;
        b =b+1;
    end
end
logrank_v2([modelset_T_low_sur_time,modelset_T_low_sur_censor],[modelset_T_high_sur_time,modelset_T_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(T_modelset,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
a = 1; b = 1;
for i =1:numel(testset(:,4))
    if (strcmp(testset{i,4},'T0') || strcmp(testset{i,4},'T1') || strcmp(testset{i,4},'T2') || strcmp(testset{i,4},'Tis'))
        testset_T_low_sur_time(a,1) = testset{i,11};
        testset_T_low_sur_censor(a,1) = testset{i,12};
        T_testset(i,1) = 0;
        a = a+1;
    elseif (strcmp(testset{i,4},'T3') || strcmp(testset{i,4},'T4'))
        testset_T_high_sur_time(b,1) = testset{i,11};
        testset_T_high_sur_censor(b,1) = testset{i,12};
        T_testset(i,1) = 1;
        b =b+1;
    end
end
logrank_v2([testset_T_low_sur_time,testset_T_low_sur_censor],[testset_T_high_sur_time,testset_T_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(T_testset,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));

%%
%临床因素分配对生存的影响，单变量分析：N,(None,N1)vs.(N2,N3)
a = 1; b = 1;
for i =1:numel(modelset(:,5))
    if (strcmp(modelset{i,5},'None') || strcmp(modelset{i,5},'N0') ||strcmp(modelset{i,5},'N1'))
        modelset_N_low_sur_time(a,1) = modelset{i,11};
        modelset_N_low_sur_censor(a,1) = modelset{i,12};
        N_modelset(i,1) = 0;
        a = a+1;
    elseif (strcmp(modelset{i,5},'N2') || strcmp(modelset{i,5},'N3'))
        modelset_N_high_sur_time(b,1) = modelset{i,11};
        modelset_N_high_sur_censor(b,1) = modelset{i,12};
        N_modelset(i,1) = 1;
        b =b+1;
    end
end
logrank_v2([modelset_N_low_sur_time,modelset_N_low_sur_censor],[modelset_N_high_sur_time,modelset_N_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(N_modelset,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
a = 1; b = 1;
for i =1:numel(testset(:,5))
    if (strcmp(testset{i,5},'None') || strcmp(testset{i,5},'N0') ||strcmp(testset{i,5},'N1'))
        testset_N_low_sur_time(a,1) = testset{i,11};
        testset_N_low_sur_censor(a,1) = testset{i,12};
        N_testset(i,1) = 0;
        a = a+1;
    elseif (strcmp(testset{i,5},'N2') || strcmp(testset{i,5},'N3'))
        testset_N_high_sur_time(b,1) = testset{i,11};
        testset_N_high_sur_censor(b,1) = testset{i,12};
        N_testset(i,1) = 1;
        b =b+1;
    end
end
logrank_v2([testset_N_low_sur_time,testset_N_low_sur_censor],[testset_N_high_sur_time,testset_N_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(N_testset,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));

%%
%临床因素分配对生存的影响，单变量分析：M,(M,M0)vs.(M1,M2)
a = 1; b = 1;
for i =1:numel(modelset(:,6))
    if (strcmp(modelset{i,6},'M') || strcmp(modelset{i,6},'M0'))
        modelset_M_low_sur_time(a,1) = modelset{i,11};
        modelset_M_low_sur_censor(a,1) = modelset{i,12};
        M_modelset(i,1) = 0;
        a = a+1;
    elseif (strcmp(modelset{i,6},'M1') || strcmp(modelset{i,6},'M2'))
        modelset_M_high_sur_time(b,1) = modelset{i,11};
        modelset_M_high_sur_censor(b,1) = modelset{i,12};
        M_modelset(i,1) = 1;
        b =b+1;
    end
end
logrank_v2([modelset_M_low_sur_time,modelset_M_low_sur_censor],[modelset_M_high_sur_time,modelset_M_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(M_modelset,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
a = 1; b = 1;
for i =1:numel(testset(:,6))
    if (strcmp(testset{i,6},'M') || strcmp(testset{i,6},'M0'))
        testset_M_low_sur_time(a,1) = testset{i,11};
        testset_M_low_sur_censor(a,1) = testset{i,12};
        M_testset(i,1) = 0;
        a = a+1;
    elseif (strcmp(testset{i,6},'M1') || strcmp(testset{i,6},'M2'))
        testset_M_high_sur_time(b,1) = testset{i,11};
        testset_M_high_sur_censor(b,1) = testset{i,12};
        M_testset(i,1) = 1;
        b =b+1;
    end
end
logrank_v2([testset_M_low_sur_time,testset_M_low_sur_censor],[testset_M_high_sur_time,testset_M_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(M_testset,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));

%%
%临床因素分配对生存的影响，单变量分析：grade,(NaN,0,1,2)vs.(3,4)
a = 1; b = 1;
for i =1:numel(modelset(:,7))
    if (strcmp(modelset{i,7},'NaN') || strcmp(modelset{i,7},'0') || strcmp(modelset{i,7},'Ⅰ') || strcmp(modelset{i,7},'Ⅱ'))
        modelset_grade_low_sur_time(a,1) = modelset{i,11};
        modelset_grade_low_sur_censor(a,1) = modelset{i,12};
        grade_modelset(i,1) = 0;
        a = a+1;
    elseif (strcmp(modelset{i,7},'Ⅲ') || strcmp(modelset{i,7},'Ⅳ'))
        modelset_grade_high_sur_time(b,1) = modelset{i,11};
        modelset_grade_high_sur_censor(b,1) = modelset{i,12};
        grade_modelset(i,1) = 1;
        b =b+1;
    end
end
logrank_v2([modelset_grade_low_sur_time,modelset_grade_low_sur_censor],[modelset_grade_high_sur_time,modelset_grade_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(grade_modelset,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
a = 1; b = 1;
for i =1:numel(testset(:,7))
    if (strcmp(testset{i,7},'NaN') || strcmp(testset{i,7},'0') || strcmp(testset{i,7},'Ⅰ') || strcmp(testset{i,7},'Ⅱ'))
        testset_grade_low_sur_time(a,1) = testset{i,11};
        testset_grade_low_sur_censor(a,1) = testset{i,12};
        grade_testset(i,1) = 0;
        a = a+1;
    elseif (strcmp(testset{i,7},'Ⅲ') || strcmp(testset{i,7},'Ⅳ'))
        testset_grade_high_sur_time(b,1) = testset{i,11};
        testset_grade_high_sur_censor(b,1) = testset{i,12};
        grade_testset(i,1) = 1;
        b =b+1;
    end
end
logrank_v2([testset_grade_low_sur_time,testset_grade_low_sur_censor],[testset_grade_high_sur_time,testset_grade_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(grade_testset,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));
%%
% X = [T_modelset N_modelset M_modelset grade_modelset predict_label_train_bestmodel];
% [b,logl,H,stats] = coxphfit(X,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
% i=1;
% for i =1: 5
%     fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p(i), exp(stats.beta(i)),exp(stats.beta(i) - 1.96*stats.se(i)),exp(stats.beta(i) + 1.96*stats.se(i)));
% end

X = [T_testset N_testset M_testset grade_testset predict_label_test_bestmodel];
[b,logl,H,stats] = coxphfit(X,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));
i=1;
for i =1: 5
    fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p(i), exp(stats.beta(i)),exp(stats.beta(i) - 1.96*stats.se(i)),exp(stats.beta(i) + 1.96*stats.se(i)));
end

[b,logl,H,stats] = coxphfit(predict_label_test_bestmodel,test_survival_time,'censoring',test_censor);
fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p, exp(stats.beta),exp(stats.beta - 1.96*stats.beta),exp(stats.beta + 1.96*stats.beta));
% [b,logl,H,stats] = coxphfit(predict_label_train_bestmodel,train_survival,'censoring',train_censor);
% [b,logl,H,stats] = coxphfit(predict_label_train_bestmodel,train_survival_time,'censoring',train_censor);
% fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p, exp(stats.beta),exp(stats.beta - 1.96*stats.beta),exp(stats.beta + 1.96*stats.beta));




