clear;
clc;

addpath('/home/ccf/matlab_project/libsvm-3.22/matlab/');
% feature_path = '/home/ccf/CCF/Colorecal-cancer/2011_survival/deep_feature_mat/';
% feature_file = dir([feature_path,'*.mat']);
[num, txt, raw] = xlsread('./survival_data/2011_3.xlsx', 'A1:L283');
load('feature_best5.mat'); load('label_281.mat');
feature = source_array' ;
clear source_array;
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
[m_0, n_0 ] = size(feature);
random_0 = randperm(m_0); 
k_0 = ceil(m_0/3);
train = feature(random_0(k_0+1:m_0),:);
train_label = label(random_0(k_0+1:m_0),:);
test = feature(random_0(1:k_0),:);
test_label = label(random_0(1:k_0),:);

%%
%特征选择 
%在模型集上面进行，利用五次交叉验证的方法进行100次迭代，找出出现次数最多的特征（100维）
feature_num = 15;
for j = 1:100
    [m, n] = size(train);
    random = randperm(m);
    k = ceil(m/5);
    data = train(random(k+1:m),:);
    label = train_label(random(k+1:m),:);
    [fea(j,:)] = mrmr_mid_d(data, label, feature_num);
end
feature_num_select = 15;
frequency = tabulate(fea(:));
frequency1 = sortrows(frequency,-2);
train_feature_select = train(:,frequency1(1:feature_num_select,1));
data_test = test(:,frequency1(1:feature_num_select,1));
%%
%分类器训练
%基于上面特征选择的方法训练分类器(SVM)，进行100次迭代和5次交叉验证评估
for i = 1:100
    random =randperm(m);
    data_train = train_feature_select(random(k+1:m),:);
    label_train = train_label(random(k+1:m),:);
    data_vail = train_feature_select(random(1:k),:);
    label_vail = train_label(random(1:k),:);
    svmmodel = svmtrain(label_train, data_train,'-b 1');
    [predict_label_vail(:,i), accuracy_vail(:,i), decision_values_vail(:,i)] = svmpredict(label_vail, data_vail, svmmodel);
    [predict_label_test(:,i), accuracy_test(:,i), decision_values_test(:,i)] = svmpredict(test_label, data_test, svmmodel);
    [predict_label_train(:,i), accuracy_train(:,i), decision_values_train(:,i)] = svmpredict(train_label, train_feature_select, svmmodel);
    [FPR_train(:,i),TPR_train(:,i), T_train(:,i), AUC_train(:,i), OPTROCPT_train(i,:),~,~] = perfcurve(train_label, predict_label_train(:,i), 1);
    [FPR_vail(:,i),TPR_vail(:,i), T_vail(:,i), AUC_vail(:,i), OPTROCPT_vail(i,:),~,~] = perfcurve(label_vail, predict_label_vail(:,i),1);
    [FPR_test(:,i),TPR_test(:,i), T_test(:,i), AUC_test(:,i), OPTROCPT_test(i,:),~,~] = perfcurve(test_label, predict_label_test(:,i), 1);
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
[predict_label_test_bestmodel, accuracy_test_bestmodel, decision_values_test_bestmodel] = svmpredict(test_label, data_test, svmmodel, '-b 1');
[predict_label_vail_bestmodel, accuracy_vail_bestmodel, decision_values_vail_bestmodel] = svmpredict(label_vail, data_vail, svmmodel, '-b 1');
[predict_label_train_bestmodel, accuracy_train_bestmodel, decision_values_train_bestmodel] = svmpredict(train_label, train_feature_select, svmmodel, '-b 1');
% [FPR_test,TPR_test, T_test, AUC_test, OPTROCPT_test,~,~] = perfcurve(test_data_label, predict_label_test_bestmodel, 1);
fprintf('AUC_vail_bestmodel: %f   AUC_test_bestmodel: %f\n',AUC_vail(:,num_bestmodel),AUC_test(:,num_bestmodel));
fprintf('accuracy_vail_bestmodel: %f   accuracy_test_bestmodel: %f\n',accuracy_vail(1,num_bestmodel), accuracy_test(1,num_bestmodel));

