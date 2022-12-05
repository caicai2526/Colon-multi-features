function para = LDA(train_data, train_label, test_data, test_label)

[a ,b] = size(train_data);
for i = 1:100
    random = randperm(a);
    data_train = train_data(random(ceil(a/5):a),:);
    label_train = train_label(random(ceil(a/5):a),:);
    data_vail = train_data(random(1:ceil(a/5)),:);
    label_vail = train_label(random(1:ceil(a/5)),:);
    LDAmodel = ClassificationDiscriminant.fit(data_train,label_train,'DiscrimType','linear');
    [predict_label_vail_LDA(:,i),decision_values_vail_LDA(:,2*i-1:2*i),stdevs_vail_LDA(:,2*i-1:2*i)]=predict(LDAmodel,data_vail);
    L = predict_label_vail_LDA(:,i) - label_vail; s = sum(~~L(:)); same = numel(L) - s; accuracy_vail_LDA(:,i) = same/numel(L);
    [predict_label_test_LDA(:,i),decision_values_test_LDA(:,2*i-1:2*i),stdevs_test_LDA(:,2*i-1:2*i)]=predict(LDAmodel,test_data);
    L = predict_label_test_LDA(:,i) - test_label; s = sum(~~L(:)); same = numel(L) - s; accuracy_test_LDA(:,i) = same/numel(L);
    save([num2str(i), '_LDAmodel', '.mat'], 'LDAmodel');
    try
        for z =2:i
            if accuracy_test_LDA(1, z-1) <= accuracy_test_LDA(1, i)
                delete([num2str(z-1),'_LDAmodel', '.mat']);
            elseif accuracy_test_LDA(1, z-1) > accuracy_test_LDA(1, i)
                delete([num2str(i),'_LDAmodel', '.mat']);
            end
        end
    end
end
LDAmodelfile = dir(['*LDAmodel.mat']);
load([LDAmodelfile(1).name]);
[predict_label_test_bestmodel_LDA, decision_values_test_bestmodel_LDA, stdevs_test_bestmodel_LDA] = predict(LDAmodel, test_data);
[predict_label_vail_bestmodel_LDA, decision_values_vail_bestmodel_LDA, stdevs_vail_bestmodel_LDA] = predict(LDAmodel, data_vail);
[predict_label_train_bestmodel_LDA ,decision_values_train_bestmodel_LDA, stdevs_train_bestmodel_LDA] = predict(LDAmodel, train_data);

[FPR_vail_LDA,TPR_vail_LDA, T_vail_LDA, AUC_vail_LDA, OPTROCPT_vail_LDA,~,~] = perfcurve(label_vail, decision_values_vail_bestmodel_LDA(:,2),1);
[FPR_test_LDA,TPR_test_LDA, T_test_LDA, AUC_test_LDA, OPTROCPT_test_LDA,~,~] = perfcurve(test_label, decision_values_test_bestmodel_LDA(:,2), 1);
[FPR_train_LDA,TPR_train_LDA, T_train_LDA, AUC_train_LDA, OPTROCPT_train_LDA,~,~] = perfcurve(train_label, decision_values_train_bestmodel_LDA(:,2), 1);


mean_test_LDA = mean(accuracy_test_LDA);
mean_vail_LDA = mean(accuracy_vail_LDA);


para.LDAmodel = LDAmodel;
para.predict_label_test_bestmodel_LDA = predict_label_test_bestmodel_LDA;
para.mean_test_LDA = mean_test_LDA;
para.mean_vail_LDA = mean_vail_LDA;
para.acc_test_LDA = max(accuracy_test_LDA);
para.FPR_test_LDA = FPR_test_LDA;
para.TPR_test_LDA = TPR_test_LDA;
para.AUC_test_LDA = AUC_test_LDA;
