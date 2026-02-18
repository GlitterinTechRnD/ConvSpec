%% Data preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xm=input';
Y=output';
Y=Y.*100;
u=mean(Xm); 
[m,nm]=size(Xm); 
for i=1:m
newdata=[Xm(i,:);u]
cov_w=cov(newdata);
dist(i)=(Xm(i,:)-u)*cov_w*(Xm(i,:)-u)'
end
[a,b]=sort(dist);
T=ceil(m*0.00005)
Threshold=a(m-T);
len=length(a);
for i = 1:len 
if a(i) < Threshold
inlier(i) = [b(i)];  
s=b(i);
disp(['Normal spectral serial number:',num2str(s)])
end
end

for i = 1:len 
if a(i)>= Threshold
outlier(i) = [b(i)];
l=b(i)
disp(['Abnormal  spectral serial number:',num2str(l)])
end
end
[c,dm]=size(inlier)
for i=1:dm
    Selected_input(i,:)=Xm(inlier(i),:); 
end
Selected_input=Selected_input';
for i=1:dm
    Selected_output(i,:)=Y(inlier(i),:); 
end
clear a;clear b;clear c;clear i;clear l;clear s;clear T;clear u;clear Y;

dataset1=Selected_input;
dataset2=sgolayfilt(dataset1,3,111);     
SNV=zscore(dataset2);                    
detrend_SNV = detrend(SNV);              

nder=1;data_diff=diff(detrend_SNV,nder); 
data_diff=sgolayfilt(data_diff,3,31);

detrend_SNV=detrend_SNV';                
t=[1:nm];
for ii=1:dm
data_diff1(ii,:)=glfdiff(detrend_SNV(ii,:),t,0.8);
end
data_diff1=data_diff1(:,nder+1:end);
data_diff1=data_diff1';
data_diff1=sgolayfilt(data_diff1,3,11); 

%% Feature Selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1=dataset1';
Y=Selected_output;
dataset=[X1 Y];
evaluat=10;
format compact
Randomiz
[o,c]=size(dataset);
disp(['objects: ' int2str(o)])
y=dataset(:,c);
v=c-1;
disp(['variables: ' int2str(v)]);
s1=[];s2=[];b=[];fin=[];sel=[];

aut=0; % autoscaling; 0=raw data; 1=column centering
ng=5; % 5 deletion groups
cr=30; % 30 chromosomes
probsel=5/v; % on average 5 variables per chromosome in the orig. pop.
maxvar=30; % 30 variables as a maximum
probmut=0.01; % probability of mutation 1%
probcross=0.5; % probability of cross-over 50%
freqb=100; % backward stepwise every 100 evaluations 
eva1=floor(evaluat/100);
eva2=evaluat/100;
if eva1==eva2;
  endb='N';
else
  endb='Y';
end
runs=100; % 100 runs
el=3;

[maxcomp,start,mxi,sxi,myi,syi]=plsgacv(dataset(:,1:v),y,aut,ng,15);
disp(' ')
disp(['With all the variables:'])
disp(['components: ' int2str(maxcomp)])
disp(['C.V. variance: ' num2str(start)])

sel=zeros(1,v); % sel stores the frequency of selection
for r=1:runs
  sel=[sel 0];
  disp(' ')
  disp(['run ' num2str(r)])
  crom=zeros(cr,v);
  resp=zeros(cr,1);
  comp=zeros(cr,1);
  p=zeros(2,v);
  numvar=zeros(cr,1); %%% numvar stores the number of variables in each chr.
  lib=[]; %%% lib is the matrix with all the already tested chromosomes %%%
  libb=[];%%% libb is the matrix with all the already backw. chromosomes %%%
  nextb=freqb;
  cc=0;
  while cc<cr
    den=0;
    sumvar=0;
    while (sumvar==0 | sumvar>maxvar)
      a=rand(1,v);
      for j=1:v
        if a(1,j)<probsel
          a(1,j)=1;
        else
          a(1,j)=0;
        end    
      end
      sumvar=sum(a);
    end
    den=CHECKTW(cc,lib,a);
    if den==0
      lib=[lib;a];
      if cc>0
        [s1,s2]=CHKSUBS(cc,crom(1:cc,:),a);
      end
      cc=cc+1;  
      var=find(a);
      [fac,risp]=plsgacv(dataset(:,var),y,aut,ng,maxcomp,mxi(:,var),sxi(:,var),myi,syi);
      if isempty(s2)
        mm=0;
      else
        mm=max(resp(s2));
      end
      if risp>mm  % the new chrom. survives only if better
        crom(cc,:)=a;
        resp(cc,1)=risp;
        comp(cc,1)=fac;
        numvar(cc,1)=size(var,2);
        for kk=1:size(s1,2)
          if risp>=resp(s1(kk))
            resp(s1(kk))=0; % the old chrom. are killed if worse
          end
        end
      end
    end
  end

  [vv,pp]=sort(resp);
  pp=flipud(pp);
  crom=crom(pp,:);
  resp=resp(pp,:);
  comp=comp(pp,:);
  numvar=numvar(pp,:);

  disp(' ')
  disp(['After the creation of the original population: ' num2str(resp(1))])
  maxrisp=resp(1);

  while cc<evaluat
    cumrisp=cumsum(resp);
    if resp(2)==0
      rr=randperm(cr);
      p(1,:)=crom(rr(1),:);
      if resp(1)==0
        p(2,:)=crom(rr(2),:);
      else
        p(2,:)=crom(1,:);
      end
    else
      k=rand*cumrisp(cr);
      j=1;
      while k>cumrisp(j)
        j=j+1;
      end
      p(1,:)=crom(j,:);
      p(2,:)=p(1,:);
      while p(2,:)==p(1,:)
        k=rand*cumrisp(cr);
        j=1;
        while k>cumrisp(j)
          j=j+1;
        end
        p(2,:)=crom(j,:);
      end
    end

    s=p;
    diff=find(p(1,:)~=p(2,:));
    randmat=rand(1,size(diff,2));
    cro=find(randmat<probcross);
    s(1,diff(cro))=p(2,diff(cro));
    s(2,diff(cro))=p(1,diff(cro));

    m=rand(2,v);
    for i=1:2
      f=find((m(i,:))<probmut);
      bb=size(f,2);
      for j=1:bb
        if s(i,f(j))==0
          s(i,f(j))=1;
        else
          s(i,f(j))=0;
        end
      end
    end
 
    for i=1:2
      den=0;
      var=find(s(i,:));
      sumvar=sum(s(i,:));
      if sumvar==0 | sumvar>maxvar
        den=1;
      end
      if den==0
        den=checktw(cc,lib,s(i,:));
      end
      if den==0
        cc=cc+1;  
	[fac,risp]=plsgacv(dataset(:,var),y,aut,ng,maxcomp,mxi(:,var),sxi(:,var),myi,syi);
        lib=[s(i,:);lib];
        if risp>maxrisp
          disp(['ev. ' int2str(cc) ' - ' num2str(risp)])
          maxrisp=risp;
        end
        if risp>resp(cr)
          [crom,resp,comp,numvar]=update(cr,crom,s(i,:),resp,comp,numvar,risp,fac,var);
        end
      end
    end

    % stepwise
    if cc>=nextb
      nextb=nextb+freqb;
      [nc,rispmax,compmax,cc,maxrisp,libb]=backw(r,cr,crom,resp,numvar,cc,dataset,y,aut,ng,maxcomp,maxrisp,libb,mxi,sxi,myi,syi,el);
      if isempty(nc)~=1
	[crom,resp,comp,numvar]=update(cr,crom,nc,resp,comp,numvar,rispmax,compmax,find(nc));
      end
    end

  end

  if endb=='Y' % final stepwise
    [nc,rispmax,compmax,cc,maxrisp,libb]=backw(r,cr,crom,resp,numvar,cc,dataset,y,aut,ng,maxcomp,maxrisp,libb,mxi,sxi,myi,syi,el);
    if isempty(nc)~=1
      [crom,resp,comp,numvar]=Update(cr,crom,nc,resp,comp,numvar,rispmax,compmax,find(nc));
    end
  end

  sel=sel(1:v)+crom(1,:);
  disp(find(crom(1,:)))
%   figure(1)
%   bar(sel);
%   set(gca,'XLim',[0 v])
%   title(['Frequency of selections after ' int2str(r) ' runs']); 
%   drawnow

end

disp('Stepwise according to the frequency of selection');
[a,b]=sort(-sel);
sel=-a;
fin=[];
k=v-1;
if v-1>200
  k=200;
end
for c=1:k
  if sel(c)>sel(c+1)
    [fac,risp]=plsgacv(dataset(:,b(1:c)),y,aut,ng,maxcomp,mxi(:,b(1:c)),sxi(:,b(1:c)),myi,syi);
    sep=sqrt(1-risp/100)*syi(ng+1);sep=sep-sep/(2*o-2); %formula "approssimata" per calcolare sep da % var. sp.
    fin=[fin [c;risp;fac;sep]];
    disp(' ')
    disp(['With ' int2str(c) ' var. ' num2str(risp) ' (' int2str(fac) ' comp.)'])
  end
end

[x,k]=max(fin(2,:));
disp(['Maximum C.V.: ' num2str(x) ' obtained with ' int2str(fin(1,k)) ' variables (' int2str(fin(3,k)) ' comp.):']);
disp(b(1:fin(1,k)))
detrend_SNV=detrend_SNV';
m=fin(1,k);
for i=1:m
    dataSelected(i,:)=dataset1(b(i),:); 
end

%% Divide the dataset & Data modeling & Data validation %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=dataset2';
X1=dataset1';
dm1=ceil(dm./3.*2);
dminmax = zeros(1,dm1); % Inicializes the vector of minimum distances.
M = size(X1,1); % Number of objects
samples = 1:M;
Dx = zeros(M,M); % Inicializes the matrix of X-distances.
Dy = zeros(M,M); % Inicializes the matriz de y-distances.
for i = 1:M-1
    xa = X1(i,:);
    ya = Y(i,:);
    for j = i + 1:M
        xb = X1(j,:);
        yb = Y(j,:);
        Dx(i, j) = norm(xa-xb);
        Dy(i, j) = norm(ya-yb);
    end
end
Dxmax = max(max(Dx));
Dymax = max(max(Dy));
D = Dx/Dxmax + Dy/Dymax; % Combines the X and y distances.
[maxD,index_row] = max(D);
[dummy,index_column] = max(maxD);
m(1) = index_row(index_column);
m(2) = index_column;
for i = 3:(dm1+1) 
    NotSelectedSample = setdiff(samples,m);
    dmin = zeros(1,M-i+1);
    for j = 1:(M-i+1)
        indexa = NotSelectedSample(j);
        d = zeros(1,i-1);
        for k = 1:(i-1)
            indexb = m(k);
            if indexa < indexb
                d(k) = D(indexa,indexb);
            else
                d(k) = D(indexb,indexa);
        end
      end
       dmin(j) = min(d);
    end

    [dummy,index] = max(dmin);
    m(i) = NotSelectedSample(index);
end
SelectedSample=setdiff(samples,NotSelectedSample);  
for i=1:length(SelectedSample)  
    XSelected(i,:)=X(SelectedSample(i),:); 
    YSelected(i,:)=Y(SelectedSample(i),:);
end  
for i=1:length(NotSelectedSample)  
    XRest(i,:)=X(NotSelectedSample(i),:); 
    YRest(i,:)=Y(NotSelectedSample(i),:); 
end  
%%%%%%%%%%%%%%%%%%  Support Vector Machine Modeling and validate  %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xin=XSelected;
Yin=YSelected;
Xt=XRest;
yt=YRest;

type = 'function estimation';  
[gam,sig2] = tunelssvm({Xin,Yin,type,[],[],'RBF_kernel'},'simplex','leaveoneoutlssvm',{'mse'});  %gam¡¢sig2 optimization
[alpha,b1] = trainlssvm({Xin,Yin,type,gam,sig2,'RBF_kernel','preprocess'});
Yt = simlssvm({Xin,Yin,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b1},Xt); 

figure;
plotlssvm({Xin,Yin,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b1});
saveas(gcf,'LSSVM','fig');

figure;
subplot(2,1,1); 
plot(Yt,'-r');
hold on;
plot(yt,'ob');
title('Testset-Comparison between true value and predicted value');ylabel('Solution concentration %');
subplot(2,1,2); 
plot(abs(Yt - yt),'-*');
title('Testset-Prediction value deviation');ylabel('Solution concentration %');

open('LSSVM.fig')
h=findall(gcf,'type','line')
x=get(h,'xdata')
y=get(h,'ydata')
y_cell=cell2mat(y);
yc=y_cell(1,:)';
Yc=y_cell(2,:)';
close(4)