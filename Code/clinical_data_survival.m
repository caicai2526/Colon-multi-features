clear;
clc;

[num, txt, raw] = xlsread('./survival_data/2011_3.xlsx', 'A1:L283');
raw(1,:) = [];
[m, n] = size(num);
random = randperm(m);
k = ceil(m/3);
modelset = raw(random(k+1:m),:);
testset = raw(random(1:k),:);
%%
%临床因素分配对生存的影响，单变量分析：年龄
age_median = median(num(:,1)); 
age_mean = mean(num(:,1));

a = 1;b = 1;
for i = 1:numel(modelset(:,3))
    try
        if modelset{i,3} >=age_median
            modelset_age_more_median_sur_time(a,1) = modelset{i,11};
            modelset_age_more_median_censor(a,1) = modelset{i,12};
            a = a+1;
        elseif modelset{i,3}<age_median
            modelset_age_less_median_sur_time(b,1) = modelset{i,11};
            modelset_age_less_median_censor(b, 1) = modelset{i,12};
            b = b+1;
        end
    end
end
logrank_v3([modelset_age_less_median_sur_time,modelset_age_less_median_censor],[modelset_age_more_median_sur_time,modelset_age_more_median_censor],0.05);
[b,logl,H,stats] = coxphfit(cell2mat(modelset(:,3)),cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
a = 1;b = 1;
for i = 1:numel(testset(:,3))
    try
        if testset{i,3} >=age_median
            testset_age_more_median_sur_time(a,1) = testset{i,11};
            testset_age_more_median_censor(a,1) = testset{i,12};
            a = a+1;
        elseif testset{i,3}<age_median
            testset_age_less_median_sur_time(b,1) = testset{i,11};
            testset_age_less_median_censor(b, 1) = testset{i,12};
            b = b+1;
        end
    end
end
logrank_v3([testset_age_less_median_sur_time,testset_age_less_median_censor],[testset_age_more_median_sur_time,testset_age_more_median_censor],0.05);
[b,logl,H,stats] = coxphfit(cell2mat(testset(:,3)),cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));

%%
%临床因素分配对生存的影响，单变量分析：性别
a = 1;b = 1;
for i =1:numel(modelset(:,2))
    if (strcmp(modelset{i,2},'male'))
        modelset_sex_male_sur_time(a,1) =  modelset{i,11};
        modelset_sex_male_censor(a,1) =  modelset{i,12};
        sex_modelset(i,1) = 1;
        a = a+1;
    elseif (strcmp(modelset{i,2},'female'))
        modelset_sex_female_sur_time(b,1) =  modelset{i,11};
        modelset_sex_female_censor(b,1) =  modelset{i,12};
        sex_modelset(i,1) = 0;
        b = b+1;
    end
end
logrank_v3([modelset_sex_male_sur_time,modelset_sex_male_censor],[modelset_sex_female_sur_time,modelset_sex_female_censor],0.05);
[b,logl,H,stats] = coxphfit(sex_modelset,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
a = 1;b = 1;
for i =1:numel(testset(:,2))
    if (strcmp(testset{i,2},'male'))
        testset_sex_male_sur_time(a,1) =  testset{i,11};
        testset_sex_male_censor(a,1) =  testset{i,12};
        sex_testset(i,1) = 1;
        a = a+1;
    elseif (strcmp(testset{i,2},'female'))
        testset_sex_female_sur_time(b,1) =  testset{i,11};
        testset_sex_female_censor(b,1) =  testset{i,12};
        sex_testset(i,1) = 0;
        b = b+1;
    end
end
logrank_v3([testset_sex_male_sur_time,testset_sex_male_censor],[testset_sex_female_sur_time,testset_sex_female_censor],0.05);
[b,logl,H,stats] = coxphfit(sex_testset,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));

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
%临床因素分配对生存的影响，单变量分析：术后是否接受化疗，接受label为1，没有接受label为0
modelset_Pos = cell2mat(modelset(:,9));
modelset_sur_time = cell2mat(modelset(:,11));
modelset_censor = cell2mat(modelset(:,12));
modelset_Pos_1_sur_time = modelset_sur_time(modelset_Pos==1);
modelset_Pos_1_censor = modelset_censor(modelset_Pos==1);
modelset_Pos_0_sur_time = modelset_sur_time(modelset_Pos==0);
modelset_Pos_0_censor = modelset_censor(modelset_Pos==0);
logrank_v3([modelset_Pos_0_sur_time,modelset_Pos_0_censor],[modelset_Pos_1_sur_time,modelset_Pos_1_censor],0.05);
[b,logl,H,stats] = coxphfit(modelset_Pos,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));

testset_Pos = cell2mat(testset(:,9));
testset_sur_time = cell2mat(testset(:,11));
testset_censor = cell2mat(testset(:,12));
testset_Pos_1_sur_time = testset_sur_time(testset_Pos==1);
testset_Pos_1_censor = testset_censor(testset_Pos==1);
testset_Pos_0_sur_time = testset_sur_time(testset_Pos==0);
testset_Pos_0_censor = testset_censor(testset_Pos==0);
logrank_v3([testset_Pos_0_sur_time,testset_Pos_0_censor],[testset_Pos_1_sur_time,testset_Pos_1_censor],0.05);
[b,logl,H,stats] = coxphfit(testset_Pos,cell2mat(testset(:,11)),'censoring',cell2mat(testset(:,12)));

%%
%临床因素分配对生存的影响,多变量进行分析
X = [T_modelset N_modelset M_modelset grade_modelset];
[b,logl,H,stats] = coxphfit(X,cell2mat(modelset(:,11)),'censoring',cell2mat(modelset(:,12)));
i=1;
for i =1: 4
    fprintf('p=% .8f, HR ratio(95CI)=% .2f(% .2f-% .2f)\n', stats.p(i), exp(stats.beta(i)),exp(stats.beta(i) - 1.96*stats.beta(i)),exp(stats.beta(i) + 1.96*stats.beta(i)));
end

