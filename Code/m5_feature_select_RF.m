clear;
clc;

addpath('/home/ccf/matlab_project/libsvm-3.22/matlab/');
feature_path = '/home/ccf/CCF/Colorecal-cancer/2011_survival/deep_feature_mat/';
feature_file = dir([feature_path,'*.mat']);
[num, txt, raw] = xlsread('./survival_data/2011.xlsx', 'A1:L285');
load('feature_label_1.mat'); load('feature_label_0.mat');

%%
%数据分配 模型集train_data, 测试验证集test_data
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
feature_num = 20;
for j = 1:100
    [m, n] = size(train_data);
    random = randperm(m);
    k = ceil(m/5);
    data = train_data(random(k+1:m),:);
    label = train_data_label(random(k+1:m),:);
    [fea(j,:)] = mrmr_mid_d(data, label, feature_num);
end
feature_num_select = 20;
frequency = tabulate(fea(:));
frequency1 = sortrows(frequency,-2);
train_feature_select = train_data(:,frequency1(1:feature_num_select,1));
data_test = test_data(:,frequency1(1:feature_num_select,1));
%%
%分类器训练
%基于上面特征选择的方法训练分类器(LAD)，进行100次迭代和5次交叉验证评估
for i = 1:100
    random =randperm(m);
    data_train = train_feature_select(random(k+1:m),:);
    label_train = train_data_label(random(k+1:m),:);
    data_vail = train_feature_select(random(1:k),:);
    label_vail = train_data_label(random(1:k),:);
    RFmodel = TreeBagger(100,data_train,label_train);
    [predict_label_vail(:,i),decision_values_vail(:,2*i-1:2*i),stdevs_vail(:,2*i-1:2*i)]=predict(RFmodel,data_vail);
    L = cellfun(@str2num,predict_label_vail(:,i)) - label_vail; s = sum(~~L(:)); same = numel(L) - s; accuracy_vail(:,i) = same/numel(L);
    %     [predict_label_vail(:,i), accuracy_vail(:,i), decision_values_vail(:,i)] = svmpredict(label_vail, data_vail, LDAmodel);
    [predict_label_test(:,i),decision_values_test(:,2*i-1:2*i),stdevs_test(:,2*i-1:2*i)]=predict(RFmodel,data_test);
    L = cellfun(@str2num,predict_label_test(:,i)) - test_data_label; s = sum(~~L(:)); same = numel(L) - s; accuracy_test(:,i) = same/numel(L);
    %     [predict_label_test(:,i), accuracy_test(:,i), decision_values_test(:,2*i-1:2*i)] = svmpredict(test_data_label, data_test, LDAmodel,'-b 1');
    [predict_label_train(:,i),decision_values_train(:,2*i-1:2*i),stdevs_train(:,2*i-1:2*i)]=predict(RFmodel,train_feature_select);
    L = cellfun(@str2num,predict_label_train(:,i)) - train_data_label; s = sum(~~L(:)); same = numel(L) - s; accuracy_train(:,i) = same/numel(L);
    %     [predict_label_train(:,i), accuracy_train(:,i), decision_values_train(:,2*i-1:2*i)] = svmpredict(train_data_label, train_feature_select, LDAmodel,'-b 1');
    [FPR_train(:,i),TPR_train(:,i), T_train(:,i), AUC_train(:,i), OPTROCPT_train(i,:),~,~] = perfcurve(train_data_label, cellfun(@str2num,predict_label_train(:,i)), 1);
    [FPR_vail(:,i),TPR_vail(:,i), T_vail(:,i), AUC_vail(:,i), OPTROCPT_vail(i,:),~,~] = perfcurve(label_vail, cellfun(@str2num,predict_label_vail(:,i)),1);
    [FPR_test(:,i),TPR_test(:,i), T_test(:,i), AUC_test(:,i), OPTROCPT_test(i,:),~,~] = perfcurve(test_data_label, cellfun(@str2num,predict_label_test(:,i)), 1);
    save([ num2str(i),'_RFmodel','.mat'], 'RFmodel');
    for z = 1:i
        if AUC_test(1,z)<AUC_test(1,i)
            delete([num2str(z),'_RFmodel','.mat']);
        elseif AUC_test(1,z)>AUC_test(1,i)
            delete([ num2str(i),'_RFmodel','.mat']);
        end
    end
end
% plot(FPR,TPR)
% xlabel('False positive rate')
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression')
%%
RFmodelfile = dir(['*RFmodel.mat']);
load([RFmodelfile(1).name]);
num_bestmodel = str2num(RFmodelfile(1).name(1:end-12));
[predict_label_test_bestmodel, decision_values_test_bestmodel, stdevs_test_bestmodel] = predict(RFmodel,data_test);
[predict_label_vail_bestmodel, decision_values_vail_bestmodel, stdevs_vail_bestmodel] = predict(RFmodel,data_vail);
[predict_label_train_bestmodel, decision_values_train_bestmodel, stdevs_train_bestmodel] = predict(RFmodel,train_feature_select);
% fprintf('AUC_vail_bestmodel: %f   AUC_test_bestmodel: %f\n',AUC_vail(:,num_bestmodel),AUC_test(:,num_bestmodel));
% fprintf('accuracy_vail_bestmodel: %f   accuracy_test_bestmodel: %f\n',accuracy_vail(:,num_bestmodel),accuracy_test(:,num_bestmodel));
% [FPR_test,TPR_test, T_test, AUC_test, OPTROCPT_test,~,~] = perfcurve(test_data_label, predict_label_test_bestmodel, 1);

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
pre_train_0 = train_survival(cellfun(@str2num,predict_label_train_bestmodel)==0);
pre_train_censor_0 = train_censor(cellfun(@str2num,predict_label_train_bestmodel)==0);
pre_train_1 = train_survival(cellfun(@str2num,predict_label_train_bestmodel)==1);
pre_train_censor_1 = train_censor(cellfun(@str2num,predict_label_train_bestmodel)==1);
pre_test_0  =test_survival(cellfun(@str2num,predict_label_test_bestmodel)==0);
pre_test_censor_0 = test_censor(cellfun(@str2num,predict_label_test_bestmodel)==0);
pre_test_1  =test_survival(cellfun(@str2num,predict_label_test_bestmodel)==1);
pre_test_censor_1 = test_censor(cellfun(@str2num,predict_label_test_bestmodel)==1);
logrank_v2([pre_train_0,pre_train_censor_0],[pre_train_1,pre_train_censor_1],0.05);
logrank_v2([pre_test_0,pre_test_censor_0],[pre_test_1,pre_test_censor_1],0.05);
fprintf('AUC_vail_bestmodel: %f   AUC_test_bestmodel: %f\n',AUC_vail(:,num_bestmodel),AUC_test(:,num_bestmodel));
fprintf('accuracy_vail_bestmodel: %f   accuracy_test_bestmodel: %f\n',accuracy_vail(:,num_bestmodel),accuracy_test(:,num_bestmodel));
% [b,logl,H,stats] = coxphfit(predict_label_test_bestmodel,test_survival,'censoring',test_censor);
% [b,logl,H,stats] = coxphfit(predict_label_train_bestmodel,train_survival,'censoring',train_censor);
