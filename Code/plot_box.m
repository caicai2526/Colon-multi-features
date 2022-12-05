clear;clc;close all;warning('off');
addpath(genpath('../../script'));
load('../class_T_noT/feature/raw_mask/feats.mat');
intra_feat = feats;
load('../class_T_noT/feature/2mm/mask_010/feats.mat');
peri_feat = feats;
load('../class_TH/feature/label_T_noT.mat');
label = label_T_noT;

intra = svm_scale(intra_feat);
peri = svm_scale(peri_feat);
selected_feat = [intra(:,899) peri(:,36) peri(:,479)... 
                 peri(:,88) peri(:,296) intra(:,64)];

%% group 1
top1_TN = selected_feat(label == 1,1);
top1_LA = selected_feat(label == 0,1);

%% group 2
top2_TN = selected_feat(label == 1,2);
top2_LA = selected_feat(label == 0,2);

%% group 3
top3_TN = selected_feat(label == 1,3);
top3_LA = selected_feat(label == 0,3);

%% group 4
top4_TN = selected_feat(label == 1,4);
top4_LA = selected_feat(label == 0,4);

%% group 5
top5_TN = selected_feat(label == 1,5);
top5_LA = selected_feat(label == 0,5);

%% group 6
top6_TN = selected_feat(label == 1,6);
top6_LA = selected_feat(label == 0,6);

X = [top1_TN; top1_LA; top2_TN; top2_LA; top3_TN; top3_LA; top4_TN; top4_LA;... 
     top5_TN; top5_LA; top6_TN; top6_LA;]; 

g1 = [ones(size(top1_TN)*2); 2*ones(size(top1_TN)*2); 3*ones(size(top1_TN)*2);...
    4*ones(size(top1_TN)*2); 5*ones(size(top1_TN)*2); 6*ones(size(top1_TN)*2);]; 
g1 = g1(:,1);
g2 = [ones(size(top1_TN)); 2*ones(size(top1_TN));];
g2 = repmat(g2,6,1);

f = figure(1)
positions = [[1:2],[5:6],[9:10],[13:14],[17:18],[21:22]];
box_plot = boxplot(X,{g1,g2},...
          'notch','off','BoxStyle','outline','whisker',1,'outliersize',4,...
          'colorgroup',g2,'symbol','.','widths',0.8,'positions',positions);

grid on
color = ['r', 'b', 'r', 'b', 'r', 'b', 'r', 'b', 'r', 'b', 'r', 'b'];
set(gca,'YLim',[-3,3],'gridLineStyle', '-.');
set(gca,'xtick',[1.5 5.5 9.5 13.5 17.5 21.5]);
set(gca,'XTickLabel',{'TN nTN '}) ;
set(box_plot,{'linew'},{1.5});
title (['top features in TN versus Non-TN']);
