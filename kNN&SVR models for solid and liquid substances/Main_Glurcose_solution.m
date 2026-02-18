clc
clear all
close all

currentDir = pwd;
disp(currentDir);
cd(currentDir);

filename = 'Glucose_data.xlsx';
data = xlsread(filename);
input = data(3:end,4:end);
output = data(1,4:end);
wavenumber = data(3:end,2);

figure;
plot(wavenumber,input); xlim([5900,8100]);
xlabel('Wavenumber(cm-1)'); ylabel('Absorptance');

run('Glurcose_solution\Modle_Glurcose.m');
test_err = yt - Yt;
train_err = yc - Yc;
n2 = length(Yt);
train_RC = corrcoef([Yc yc]);
test_RP = corrcoef([Yt yt]);
test_RMSE = sqrt(sum((test_err).^2)/n2);
train_RMSE = sqrt(sum((train_err).^2)/n2);
test_MAE=mean(abs(test_err));
average=sum(abs(yt-Yt))./n2;
R=[average./yt];
Rave=sum(R)./n2;

figure;
RC=train_RC(2,1);
RP=test_RP(2,1);
dm2=length(Yt);
p=fittype('poly1')
f=fit(yt,Yt,p)
plot(f,yt,Yt,'o');
hold on
scatter(yc,Yc,'*');
hold off
grid on
xlabel('Actual concentration (%)');
ylabel('Predicted concentration (%)');
title('Modeling and Testing');
text( 'string',['Rc^2 = ',num2str(RC*RC)],'Units','normalized','position',[0.05,0.95]);
text( 'string',['Rp^2 = ',num2str(RP*RP)],'Units','normalized','position',[0.05,0.9]);
text( 'string',['RMSE = ',num2str(test_RMSE),' %'],'Units','normalized','position',[0.05,0.8]);
text( 'string',['MAE = ',num2str(test_MAE),' %'],'Units','normalized','position',[0.05,0.85]);