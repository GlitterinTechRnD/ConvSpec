function [X,y]=simuin(Mx,Px,Nnoise,nLevel)
%+++ SIMUIN simulation based on UVE paper.


if nargin<4;nLevel=0.005;end
if nargin<3;Nnoise=0;end
if nargin<2;Px=100;end
if nargin<1;Mx=25;end
    
method='center';
x=rand(Mx,Px);
x=pretreat(x,'center');
[u,s,v]=svd(x);
d=diag(s);
d=d(1:5)/sum(d);
T=x*v(:,1:5);
X0=T*v(:,1:5)';
noise=randn(Mx,Px+Nnoise);
noise=nLevel*noise/max(max(abs(noise)));
X=[X0 rand(Mx,Nnoise)]+noise;  % SIMUIN
y=T*[5 4 3 2 1]';







