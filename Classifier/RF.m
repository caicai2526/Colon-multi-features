function para = RF(train_data, train_label, test_data, test_label)

[a ,b] = size(train_data);

for i = 1:100
    random = randperm(a);
    data_train = train_data(random(ceil(a/5):a),:);
    label_train = train_label(random(ceil(a/5):a),:);
    data_vail = train_data(random(1:ceil(a/5)),:);
    label_vail = train_label(random(1:ceil(a/5)),:);
    RFmodel = TreeBagger(100,data_train,label_train);
    [predict_label_vail_RF(:,i),decision_values_vail_RF(:,2*i-1:2*i),stdevs_vail_RF(:,2*i-1:2*i)]=predict(RFmodel,data_vail);
    L = cellfun(@str2num,predict_label_vail_RF(:,i)) - label_vail; s = sum(~~L(:)); same = numel(L) - s; accuracy_vail_RF(:,i) = same/numel(L);
    [predict_label_test_RF(:,i),decision_values_test_RF(:,2*i-1:2*i),stdevs_test_RF(:,2*i-1:2*i)]=predict(RFmodel,test_data);
    L = cellfun(@str2num,predict_label_test_RF(:,i)) - test_label; s = sum(~~L(:)); same = numel(L) - s; accuracy_test_RF(:,i) = same/numel(L);
    save([num2str(i), '_RFmodel', '.mat'], 'RFmodel');
    try
        for z =2:i
            if accuracy_test_RF(1, z-1) <= accuracy_test_RF(1, i)
                delete([num2str(z-1),'_RFmodel', '.mat']);
            elseif accuracy_test_RF(1, z-1) > accuracy_test_RF(1, i)
                delete([num2str(i),'_RFmodel', '.mat']);
            end
        end
    end
end
RFmodelfile = dir(['*RFmodel.mat']);
load([RFmodelfile(1).name]);
[predict_label_test_bestmodel_RF, decision_values_test_bestmodel_RF, stdevs_test_bestmodel_RF] = predict(RFmodel, test_data);
[predict_label_vail_bestmodel_RF, decision_values_vail_bestmodel_RF, stdevs_vail_bestmodel_RF] = predict(RFmodel, data_vail);
[predict_label_train_bestmodel_RF ,decision_values_train_bestmodel_RF, stdevs_train_bestmodel_RF] = predict(RFmodel, train_data);

[FPR_vail_RF,TPR_vail_RF, T_vail_RF, AUC_vail_RF, OPTROCPT_vail_RF,~,~] = perfcurve(label_vail, decision_values_vail_bestmodel_RF(:,2),1);
[FPR_test_RF,TPR_test_RF, T_test_RF, AUC_test_RF, OPTROCPT_test_RF,~,~] = perfcurve(test_label, decision_values_test_bestmodel_RF(:,2), 1);
[FPR_train_RF,TPR_train_RF, T_train_RF, AUC_train_RF, OPTROCPT_train_RF,~,~] = perfcurve(train_label, decision_values_train_bestmodel_RF(:,2), 1);

mean_test_RF = mean(accuracy_test_RF);
mean_vail_RF = mean(accuracy_vail_RF);

para.RFmodel = RFmodel;
para.predict_label_test_bestmodel_RF = predict_label_test_bestmodel_RF;
para.mean_test_RF = mean_test_RF;
para.mean_vail_RF = mean_vail_RF;
para.acc_test_RF = max(accuracy_test_RF);
para.FPR_test_RF = FPR_test_RF;
para.TPR_test_RF = TPR_test_RF;
para.AUC_test_RF = AUC_test_RF;