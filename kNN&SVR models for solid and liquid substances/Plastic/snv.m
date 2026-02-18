function [X_snv] = snv(X)
% Standard Normal Variate
%
% [x_snv] = snv(x) 
%
% input:
% x (samples x variables) data to preprocess
%
% output:
% x_snv (samples x variables) preprocessed data
%
% By Cleiton A. Nunes
% UFLA,MG,Brazil

[m,n]=size(X);
rmean=mean(X,2);
dr=X-repmat(rmean,1,n);
X_snv=dr./repmat(sqrt(sum(dr.^2,2)/(n-1)),1,n);