function para = SVM(train_data, train_label, test_data, test_label)

[a ,b] = size(train_data);
for i = 1:100
    random = randperm(a);
    data_train = train_data(random(ceil(a/5):a),:);
    label_train = train_label(random(ceil(a/5):a),:);
    data_vail = train_data(random(1:ceil(a/5)),:);
    label_vail = train_label(random(1:ceil(a/5)),:);
    SVMmodel = svmtrain(label_train, data_train,'-b 1');
    [predict_label_vail_SVM(:,i), accuracy_vail_SVM(:,i), decision_values_vail_SVM(:,i)] = svmpredict(label_vail, data_vail, SVMmodel);
    [predict_label_test_SVM(:,i), accuracy_test_SVM(:,i), decision_values_test_SVM(:,i)] = svmpredict(test_label, test_data, SVMmodel);
    save([num2str(i), '_SVMmodel', '.mat'], 'SVMmodel');
    try
        for z =2:i
            if accuracy_test_SVM(1, z-1) <= accuracy_test_SVM(1, i)
                delete([num2str(z-1),'_SVMmodel', '.mat']);
            elseif accuracy_test_SVM(1, z-1) > accuracy_test_SVM(1, i)
                delete([num2str(i),'_SVMmodel', '.mat']);
            end
        end
    end
end
SVMmodelfile = dir(['*SVMmodel.mat']);
load([SVMmodelfile(1).name]);
[predict_label_test_bestmodel_SVM, accuracy_test_bestmodel_SVM,decision_values_test_bestmodel_SVM] = svmpredict(test_label, test_data,SVMmodel, '-b 1');
[predict_label_vail_bestmodel_SVM, accuracy_vail_bestmodel_SVM,decision_values_vail_bestmodel_SVM] = svmpredict(label_vail, data_vail,SVMmodel, '-b 1');
[predict_label_train_bestmodel_SVM, accuracy_train_bestmodel_SVM,decision_values_train_bestmodel_SVM] = svmpredict(train_label, train_data,SVMmodel, '-b 1');

[FPR_vail_SVM,TPR_vail_SVM, T_vail_SVM, AUC_vail_SVM, OPTROCPT_vail_SVM,~,~] = perfcurve(label_vail, decision_values_vail_bestmodel_SVM(:,1),1);
[FPR_test_SVM,TPR_test_SVM, T_test_SVM, AUC_test_SVM, OPTROCPT_test_SVM,~,~] = perfcurve(test_label, decision_values_test_bestmodel_SVM(:,1), 1);
[FPR_train_SVM,TPR_train_SVM, T_train_SVM, AUC_train_SVM, OPTROCPT_train_SVM,~,~] = perfcurve(train_label, decision_values_train_bestmodel_SVM(:,1), 1);

mean_test_SVM = mean(accuracy_test_SVM(1,:));
mean_vail_SVM = mean(accuracy_vail_SVM(1,:));

para.SVMmodel = SVMmodel;
para.predict_label_test_bestmodel_SVM = predict_label_test_bestmodel_SVM;
para.mean_test_SVM = mean_test_SVM;
para.mean_vail_SVM = mean_vail_SVM;
para.acc_test_SVM = max(accuracy_test_SVM(1,:));
para.FPR_test_SVM = FPR_test_SVM;
para.TPR_test_SVM = TPR_test_SVM;
para.AUC_test_SVM = AUC_test_SVM;