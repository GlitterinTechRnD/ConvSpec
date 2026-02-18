function LS=lsreg(X,y,methodx,methody)
%+++ Ordinary Least Squares regression;
%+++ Advisor: Yizeng Liang, yizeng_liang@263.net
%+++ H.D. Li,lhdcsu@gmail.com

if nargin<4; methody='center';end;
if nargin<4; methodx='center';end;

n=length(y);

%+++ Pretreatment
[Xs,xpara1,xpara2]=pretreat(X,methodx);
[ys,ypara1,ypara2]=pretreat(y,methody);

%+++ standardized coefficent of Least squares regression 
beta=pinv(Xs)*ys;

%+++ Back to original variable space;
beta1=ypara2/xpara2*beta;
beta0=ypara1-beta1*xpara1;

%+++ Fitting
yPred=beta1*(X-xpara1)+ypara1;

%+++ Compute prediction error
residual=yPred-y;
s=sqrt(sumsqr(residual)/(n-2));
Sxx=sumsqr(X-xpara1);

%+++ Compute the standard error of regression coefficients
SDbeta1=sqrt(s^2/Sxx);
SDbeta0=sqrt(s^2*(1/n+xpara1^2/Sxx));

%+++ Output
LS.beta0=beta0;
LS.beta1=beta1;
LS.yPred=yPred;
LS.residual=residual;
LS.sigma=s;
LS.SDbeta0=SDbeta0;
LS.SDbeta1=SDbeta1;





