function para = KNN(train_data, train_label, test_data, test_label)

[a ,b] = size(train_data);

for i = 1:100
    random = randperm(a);
    data_train = train_data(random(ceil(a/5):a),:);
    label_train = train_label(random(ceil(a/5):a),:);
    data_vail = train_data(random(1:ceil(a/5)),:);
    label_vail = train_label(random(1:ceil(a/5)),:);
    KNNmodel = fitcknn(data_train,label_train,'NumNeighbors',5);
    [predict_label_vail_KNN(:,i),decision_values_vail_KNN(:,2*i-1:2*i),stdevs_vail_KNN(:,2*i-1:2*i)]=predict(KNNmodel,data_vail);
    L = predict_label_vail_KNN(:,i) - label_vail; s = sum(~~L(:)); same = numel(L) - s; accuracy_vail_KNN(:,i) = same/numel(L);
    [predict_label_test_KNN(:,i),decision_values_test_KNN(:,2*i-1:2*i),stdevs_test_KNN(:,2*i-1:2*i)]=predict(KNNmodel,test_data);
    L = predict_label_test_KNN(:,i) - test_label; s = sum(~~L(:)); same = numel(L) - s; accuracy_test_KNN(:,i) = same/numel(L);
    save([num2str(i), '_KNNmodel', '.mat'], 'KNNmodel');
    try
        for z =2:i
            if accuracy_test_KNN(1, z-1) <= accuracy_test_KNN(1, i)
                delete([num2str(z-1),'_KNNmodel', '.mat']);
            elseif accuracy_test_KNN(1, z-1) > accuracy_test_KNN(1, i)
                delete([num2str(i),'_KNNmodel', '.mat']);
            end
        end
    end
end
KNNmodelfile = dir(['*KNNmodel.mat']);
load([KNNmodelfile(1).name]);
[predict_label_test_bestmodel_KNN, decision_values_test_bestmodel_KNN, stdevs_test_bestmodel_KNN] = predict(KNNmodel, test_data);
[predict_label_vail_bestmodel_KNN, decision_values_vail_bestmodel_KNN, stdevs_vail_bestmodel_KNN] = predict(KNNmodel, data_vail);
[predict_label_train_bestmodel_KNN ,decision_values_train_bestmodel_KNN, stdevs_train_bestmodel_KNN] = predict(KNNmodel, train_data);

[FPR_vail_KNN,TPR_vail_KNN, T_vail_KNN, AUC_vail_KNN, OPTROCPT_vail_KNN,~,~] = perfcurve(label_vail, decision_values_vail_bestmodel_KNN(:,2),1);
[FPR_test_KNN,TPR_test_KNN, T_test_KNN, AUC_test_KNN, OPTROCPT_test_KNN,~,~] = perfcurve(test_label, decision_values_test_bestmodel_KNN(:,2), 1);
[FPR_train_KNN,TPR_train_KNN, T_train_KNN, AUC_train_KNN, OPTROCPT_train_KNN,~,~] = perfcurve(train_label, decision_values_train_bestmodel_KNN(:,2), 1);

mean_test_KNN = mean(accuracy_test_KNN);
mean_vail_KNN = mean(accuracy_vail_KNN);

para.KNNmodel = KNNmodel;
para.predict_label_test_bestmodel_KNN = predict_label_test_bestmodel_KNN;
para.mean_test_KNN = mean_test_KNN;
para.mean_vail_KNN = mean_vail_KNN;
para.acc_test_KNN = max(accuracy_test_KNN);
para.FPR_test_KNN = FPR_test_KNN;
para.TPR_test_KNN = TPR_test_KNN;
para.AUC_test_KNN = AUC_test_KNN;