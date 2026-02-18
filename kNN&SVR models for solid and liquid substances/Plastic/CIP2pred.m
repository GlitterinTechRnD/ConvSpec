function F=CIP2pred(X,Y,B,Q,N,A,criterion,fold)
%+++ test the relationship between the variable combination score and its
%    prediction error.

if nargin<8;fold=5;end
if nargin<7;criterion=0;end
if nargin<6;A=5;end
if nargin<5;N=1000;end

B=abs(B);
p=length(B);
C=[];
C=corrcoef(X);
G=computeCIPMap(B,criterion,C);

K=size(nchoosek(1:Q,2),1);
predError=nan(N,1);
score=nan(N,1);
Q2=nan(N,1);

for i=1:N
    v=randperm(p);
    v=v(1:Q);
    score(i)=computeCIP(G,v);
    CV=plscvfold(X(:,v),Y,A,fold,'autoscaling',0);
    predError(i)=CV.RMSECV;
    Q2(i)=CV.Q2_max;
    if rem(i,100)==0;fprintf('The %d/%dth sampling for %d variables finished.\n',i,N,Q);end
end

%+++ output
F.score=score;
F.predError=predError;
F.Q2=Q2;


    




