function F=variableSetComparion(X,Y,V1,V2,A,K,method,ratio,N)
%+++ Model populatin analysis for statistical model(variabel sets here) comparison
%+++ input:
%    

%+++

%+++ Default
if nargin<9;N=1000;end
if nargin<8;ratio=0.8;end
if nargin<7;method='autoscaling';end
if nargin<6;K=5;;end
if nargin<5;A=10;end


[m,n]=size(X);
lv(1)=length(V1);
lv(2)=length(V2);
V{1}=V1;
V{2}=V2;
D=nan(N,1);
A=min([n A lv]);

for i=1:N
  [Xcal,Ycal,Xtest,Ytest]=traintestselect(X,Y,ratio,0);  
  for j=1:2
    Xtemp=Xcal(:,V{j});  
    CV=plscvfold(Xtemp,Ycal,A,K,method,0);
    PLS=pls(Xtemp,Ycal,CV.optPC,method);
    [ypred,RMSEPtemp]=plsval(PLS,Xtest(:,V{j}),Ytest);
    RMSEP(i,j)=RMSEPtemp;
  end
  D(i)=RMSEP(i,1)-RMSEP(i,2);
  fprintf('The %dth iteration finished.\n',i);
end

%+++ t-test
[h,p] = ttest(D);
winner=1;
if mean(D)>0;winner=2;end

%+++ output
F.ratio=ratio;
F.N=N;
F.RMSEP=RMSEP;
F.D=D;
F.reject=h;
F.p=p;
F.winner=winner;



