function para = QDA(train_data, train_label, test_data, test_label)

[a ,b] = size(train_data);
 for i = 1:100
    random = randperm(a);
    data_train = train_data(random(ceil(a/5):a),:);
    label_train = train_label(random(ceil(a/5):a),:);
    data_vail = train_data(random(1:ceil(a/5)),:);
    label_vail = train_label(random(1:ceil(a/5)),:);
    QDAmodel = ClassificationDiscriminant.fit(data_train,label_train,'DiscrimType','quadratic');
    [predict_label_vail_QDA(:,i),decision_values_vail_QDA(:,2*i-1:2*i),stdevs_vail_QDA(:,2*i-1:2*i)]=predict(QDAmodel,data_vail);
    L = predict_label_vail_QDA(:,i) - label_vail; s = sum(~~L(:)); same = numel(L) - s; accuracy_vail_QDA(:,i) = same/numel(L);
    [predict_label_test_QDA(:,i),decision_values_test_QDA(:,2*i-1:2*i),stdevs_test_QDA(:,2*i-1:2*i)]=predict(QDAmodel,test_data);
    L = predict_label_test_QDA(:,i) - test_label; s = sum(~~L(:)); same = numel(L) - s; accuracy_test_QDA(:,i) = same/numel(L);
    save([num2str(i), '_QDAmodel', '.mat'], 'QDAmodel');
    try
        for z =2:i
            if accuracy_test_QDA(1, z-1) <= accuracy_test_QDA(1, i)
                delete([num2str(z-1),'_QDAmodel', '.mat']);
            elseif accuracy_test_QDA(1, z-1) > accuracy_test_QDA(1, i)
                delete([num2str(i),'_QDAmodel', '.mat']);
            end
        end
    end
end
QDAmodelfile = dir(['*QDAmodel.mat']);
load([QDAmodelfile(1).name]);
[predict_label_test_bestmodel_QDA, decision_values_test_bestmodel_QDA, stdevs_test_bestmodel_QDA] = predict(QDAmodel, test_data);
[predict_label_vail_bestmodel_QDA, decision_values_vail_bestmodel_QDA, stdevs_vail_bestmodel_QDA] = predict(QDAmodel, data_vail);
[predict_label_train_bestmodel_QDA ,decision_values_train_bestmodel_QDA, stdevs_train_bestmodel_QDA] = predict(QDAmodel, train_data);

[FPR_vail_QDA,TPR_vail_QDA, T_vail_QDA, AUC_vail_QDA, OPTROCPT_vail_QDA,~,~] = perfcurve(label_vail, decision_values_vail_bestmodel_QDA(:,2),1);
[FPR_test_QDA,TPR_test_QDA, T_test_QDA, AUC_test_QDA, OPTROCPT_test_QDA,~,~] = perfcurve(test_label, decision_values_test_bestmodel_QDA(:,2), 1);
[FPR_train_QDA,TPR_train_QDA, T_train_QDA, AUC_train_QDA, OPTROCPT_train_QDA,~,~] = perfcurve(train_label, decision_values_train_bestmodel_QDA(:,2), 1);

mean_test_QDA = mean(accuracy_test_QDA);
mean_vail_QDA = mean(accuracy_vail_QDA);

para.QDAmodel = QDAmodel;
para.predict_label_test_bestmodel_QDA = predict_label_test_bestmodel_QDA;
para.mean_test_QDA = mean_test_QDA;
para.mean_vail_QDA = mean_vail_QDA;
para.acc_test_QDA = max(accuracy_test_QDA);
para.FPR_test_QDA = FPR_test_QDA;
para.TPR_test_QDA = TPR_test_QDA;
para.AUC_test_QDA = AUC_test_QDA;