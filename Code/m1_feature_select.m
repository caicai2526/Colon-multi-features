clear;
clc;

addpath('/home/ccf/matlab_project/libsvm-3.22/matlab/');
feature_path = '/home/ccf/CCF/Colorecal-cancer/2011_survival/deep_feature_mat/';
feature_file = dir([feature_path,'*.mat']);
[num, txt, raw] = xlsread('./survival_data/2011_3.xlsx', 'A1:L283');
load('feature_label_1.mat'); load('feature_label_0.mat');
% feature_label_1 = []; feature_label_0  =[];a = 1;b = 1;
% for i = 1 : numel(num(:,9))
%     try
%         if num(i, 9)>=60
%             name = raw{i+1,1};
%             load([feature_path,name,'_100000.mat']);
%             feature_label_1 = [feature_label_1 label];
%             survival_1(a,1) = num(i,9);
%             a = a+1;
%         elseif num(i,9)<60
%             name = raw{i+1,1};
%             load([feature_path,name,'_100000.mat']);
%             feature_label_0 = [feature_label_0 label];
%             survival_0(b,1) = num(i,9);
%             b = b+1;
%         end
%     end
% end
%%
%数据分配 模型集train_data, 测试验证集test_data
% feature_label_0 = feature_label_0';
% feature_label_1 = feature_label_1';
% feature = [feature_label_0;feature_label_1];
% label = [zeros(m_0,1);ones(m_1,1)];
[m_0, n_0 ] = size(feature_label_0);
random_0 = randperm(m_0); 
k_0 = ceil(m_0/3);
train_label_0 = feature_label_0(random_0(k_0+1:m_0),:);
label_train_0 = zeros(m_0-k_0,1);
test_label_0 = feature_label_0(random_0(1:k_0),:);
label_test_0 = zeros(k_0,1);
[m_1, n_1 ] = size(feature_label_1);
random_1 = randperm(m_1); 
k_1 = ceil(m_1/3);
train_label_1 = feature_label_1(random_1(k_1+1:m_1),:);
label_train_1 = ones(m_1-k_1,1);
test_label_1 = feature_label_1(random_1(1:k_1),:);
label_test_1 = ones(k_1,1);
train_data = [train_label_0;train_label_1];
train_data_label = [label_train_0;label_train_1];
test_data = [test_label_0;test_label_1];
test_data_label = [label_test_0;label_test_1];
%%
%特征选择 
%在模型集上面进行，利用五次交叉验证的方法进行100次迭代，找出出现次数最多的特征（100维）
feature_num = 15;
for j = 1:100
    [m, n] = size(train_data);
    random = randperm(m);
    k = ceil(m/5);
    data = train_data(random(k+1:m),:);
    label = train_data_label(random(k+1:m),:);
    [fea(j,:)] = mrmr_mid_d(data, label, feature_num);
end
feature_num_select = 15;
frequency = tabulate(fea(:));
frequency1 = sortrows(frequency,-2);
train_feature_select = train_data(:,frequency1(1:feature_num_select,1));
data_test = test_data(:,frequency1(1:feature_num_select,1));
%%
%分类器训练
%基于上面特征选择的方法训练分类器(SVM)，进行100次迭代和5次交叉验证评估
for i = 1:100
    random =randperm(m);
    data_train = train_feature_select(random(k+1:m),:);
    label_train = train_data_label(random(k+1:m),:);
    data_vail = train_feature_select(random(1:k),:);
    label_vail = train_data_label(random(1:k),:);
    svmmodel = svmtrain(label_train, data_train,'-b 1');
    [predict_label_vail(:,i), accuracy_vail(:,i), decision_values_vail(:,i)] = svmpredict(label_vail, data_vail, svmmodel);
    [predict_label_test(:,i), accuracy_test(:,i), decision_values_test(:,2*i-1:2*i)] = svmpredict(test_data_label, data_test, svmmodel,'-b 1');
    [predict_label_train(:,i), accuracy_train(:,i), decision_values_train(:,2*i-1:2*i)] = svmpredict(train_data_label, train_feature_select, svmmodel,'-b 1');
    [FPR_train(:,i),TPR_train(:,i), T_train(:,i), AUC_train(:,i), OPTROCPT_train(i,:),~,~] = perfcurve(train_data_label, predict_label_train(:,i), 1);
    [FPR_vail(:,i),TPR_vail(:,i), T_vail(:,i), AUC_vail(:,i), OPTROCPT_vail(i,:),~,~] = perfcurve(label_vail, predict_label_vail(:,i),1);
    [FPR_test(:,i),TPR_test(:,i), T_test(:,i), AUC_test(:,i), OPTROCPT_test(i,:),~,~] = perfcurve(test_data_label, predict_label_test(:,i), 1);
    save([ num2str(i),'_svmmodel','.mat'], 'svmmodel');
    for z = 1:i
        if accuracy_vail(1,z)<accuracy_vail(1,i)
            delete([num2str(z),'_svmmodel','.mat']);
        elseif accuracy_vail(1,z)>accuracy_vail(1,i)
            delete([ num2str(i),'_svmmodel','.mat']);
        end
    end
end
% plot(FPR,TPR)
% xlabel('False positive rate')
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression')
%%
%测试测试集数据
svmmodelfile = dir(['*svmmodel.mat']);
load([svmmodelfile(1).name]);
num_bestmodel = str2num(svmmodelfile(1).name(1:end-13));
[predict_label_test_bestmodel, accuracy_test_bestmodel, decision_values_test_bestmodel] = svmpredict(test_data_label, data_test, svmmodel, '-b 1');
[predict_label_vail_bestmodel, accuracy_vail_bestmodel, decision_values_vail_bestmodel] = svmpredict(label_vail, data_vail, svmmodel, '-b 1');
[predict_label_train_bestmodel, accuracy_train_bestmodel, decision_values_train_bestmodel] = svmpredict(train_data_label, train_feature_select, svmmodel, '-b 1');
% [FPR_test,TPR_test, T_test, AUC_test, OPTROCPT_test,~,~] = perfcurve(test_data_label, predict_label_test_bestmodel, 1);
fprintf('AUC_vail_bestmodel: %f   AUC_test_bestmodel: %f\n',AUC_vail(:,num_bestmodel),AUC_test(:,num_bestmodel));
fprintf('accuracy_vail_bestmodel: %f   accuracy_test_bestmodel: %f\n',accuracy_vail(1,num_bestmodel), accuracy_test(1,num_bestmodel));

%%
%生存分析
a = 1;b = 1;
for i = 1 : numel(num(:,9))
    try
        if num(i, 9)>=60
            survival_1(a,1) = num(i,9);
            censor_1(a,1) = num(i,10);
            a = a+1;
        elseif num(i,9)<60
            survival_0(b,1) = num(i,9);
            censor_0(b,1) = num(i, 10);
            b = b+1;
        end
    end
end
train_survival_0 = survival_0(random_0(k_0+1:m_0),:);
test_survival_0 = survival_0(random_0(1:k_0),:);
train_survival_1 = survival_1(random_1(k_1+1:m_1),:);
test_survival_1 = survival_1(random_1(1:k_1),:);
train_censor_0 = censor_0(random_0(k_0+1:m_0),:);
test_censor_0 = censor_0(random_0(1:k_0),:);
train_censor_1 = censor_1(random_1(k_1+1:m_1),:);
test_censor_1= censor_1(random_1(1:k_1),:);
train_survival = [train_survival_0;train_survival_1];
test_survival = [test_survival_0;test_survival_1];
train_censor = [train_censor_0;train_censor_1];
test_censor = [test_censor_0;test_censor_1];
pre_train_0 = train_survival(predict_label_train_bestmodel==0);
pre_train_censor_0 = train_censor(predict_label_train_bestmodel==0);
pre_train_1 = train_survival(predict_label_train_bestmodel==1);
pre_train_censor_1 = train_censor(predict_label_train_bestmodel==1);
pre_test_0  =test_survival(predict_label_test_bestmodel==0);
pre_test_censor_0 = test_censor(predict_label_test_bestmodel==0);
pre_test_1  =test_survival(predict_label_test_bestmodel==1);
pre_test_censor_1 = test_censor(predict_label_test_bestmodel==1);
logrank_v2([pre_train_0,pre_train_censor_0],[pre_train_1,pre_train_censor_1],0.05);
logrank_v2([pre_test_0,pre_test_censor_0],[pre_test_1,pre_test_censor_1],0.05);

%%
a = 1;b = 1;
for i = 1 : numel(num(:,9))
    try
        if num(i, 9)>=60
            raw_1(a,:) = raw(i+1,:);
            survival_1(a,1) = num(i,9);
            censor_1(a,1) = num(i,10);
            a = a+1;
        elseif num(i,9)<60
            raw_0(b,:) = raw(i+1,:);
            survival_0(b,1) = num(i,9);
            censor_0(b,1) = num(i, 10);
            b = b+1;
        end
    end
end
modelset_0 = raw_0(random_0(k_0+1:m_0),:);
testset_0 = raw_0(random_0(1:k_0),:);
modelset_1 =raw_1(random_1(k_1+1:m_1),:);
testset_1 = raw_1(random_1(1:k_1),:);
modelset = [modelset_0;modelset_1];
testset =[testset_0;testset_1];
%%
%临床因素分配对生存的影响，单变量分析：T,(T0,Tis,T1,T2)vs.(T3,T4)
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
logrank_v3([modelset_T_low_sur_time,modelset_T_low_sur_censor],[modelset_T_high_sur_time,modelset_T_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(T_modelset,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
a = 1; b = 2;
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
logrank_v3([testset_T_low_sur_time,testset_T_low_sur_censor],[testset_T_high_sur_time,testset_T_high_sur_censor],0.05);
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
logrank_v3([modelset_N_low_sur_time,modelset_N_low_sur_censor],[modelset_N_high_sur_time,modelset_N_high_sur_censor],0.05);
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
logrank_v3([testset_N_low_sur_time,testset_N_low_sur_censor],[testset_N_high_sur_time,testset_N_high_sur_censor],0.05);
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
logrank_v3([modelset_M_low_sur_time,modelset_M_low_sur_censor],[modelset_M_high_sur_time,modelset_M_high_sur_censor],0.05);
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
logrank_v3([testset_M_low_sur_time,testset_M_low_sur_censor],[testset_M_high_sur_time,testset_M_high_sur_censor],0.05);
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
logrank_v3([modelset_grade_low_sur_time,modelset_grade_low_sur_censor],[modelset_grade_high_sur_time,modelset_grade_high_sur_censor],0.05);
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
logrank_v3([testset_grade_low_sur_time,testset_grade_low_sur_censor],[testset_grade_high_sur_time,testset_grade_high_sur_censor],0.05);
[b,logl,H,stats] = coxphfit(grade_testset,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));
%%
X = [T_modelset N_modelset M_modelset grade_modelset predict_label_train_bestmodel];
[b,logl,H,stats] = coxphfit(X,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
i=1;
for i =1: 5
    fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p(i), exp(stats.beta(i)),exp(stats.beta(i) - 1.96*stats.se(i)),exp(stats.beta(i) + 1.96*stats.se(i)));
end

X = [T_testset N_testset M_testset grade_testset predict_label_test_bestmodel];
[b,logl,H,stats] = coxphfit(X,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));
i=1;
for i =1: 5
    fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p(i), exp(stats.beta(i)),exp(stats.beta(i) - 1.96*stats.se(i)),exp(stats.beta(i) + 1.96*stats.se(i)));
end

% [b,logl,H,stats] = coxphfit(predict_label_test_bestmodel,test_survival,'censoring',test_censor);
% fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p, exp(stats.beta),exp(stats.beta - 1.96*stats.beta),exp(stats.beta + 1.96*stats.beta));
% % [b,logl,H,stats] = coxphfit(predict_label_train_bestmodel,train_survival,'censoring',train_censor);
% [b,logl,H,stats] = coxphfit(predict_label_train_bestmodel,train_survival,'censoring',train_censor);
% fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p, exp(stats.beta),exp(stats.beta - 1.96*stats.beta),exp(stats.beta + 1.96*stats.beta));

