function [ idx ] = KNN( trainData,trainClass,testData,K )
%   UNTITLED Summary of this function goes here
%   Detailed explanation goes here
 
 
[N,M]=size(trainData);
%计算训练数据集与测试数据之间的欧氏距离dist
dist=zeros(N,1);
for i=1:N
    dist(i,:)=norm(trainData(i,:)-testData);
end
%将dist从小到大进行排序
[Y,I]=sort(dist,1);   
K=min(K,length(Y));
%将训练数据对应的类别与训练数据排序结果对应
labels=trainClass(I);
%{
%确定前K个点所在类别的出现频率
classNum=length(unique(trainClass));%取集合中的单值元素的个数
labels=zeros(1,classNum);
for i=1:K
    j=trainClass(i);
    labels(j)=labels(j)+1;
end
%返回前K个点中出现频率最高的类别作为测试数据的预测分类
[~,idx]=max(labels);
%}
%确定前K个点所在类别的出现频率
idx=mode(labels(1:K));%mode函数求众数
fprintf('%d  ',idx);
end

