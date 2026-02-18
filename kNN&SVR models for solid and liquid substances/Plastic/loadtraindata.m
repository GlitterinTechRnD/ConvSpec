function [train_x,train_y]=loadtraindata()
train_x=importdata('F:\安装包\LSSVMlabv1_8_R2009b_R2011a\INPUT.mat');
train_y=importdata('F:\安装包\LSSVMlabv1_8_R2009b_R2011a\OUTPUT.mat');
end