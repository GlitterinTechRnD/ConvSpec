function v=rswrw(n,k,w)
%+++ weighted random sampling without replacement 

v=zeros(1,k);
for i=1:k
    v(i)=randsample(n,1,true,w);
    w(v(i))=0;
    w=w./sum(w);
end
