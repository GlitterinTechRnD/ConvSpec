clc
clear all
close all

currentDir = pwd;
disp(currentDir);
cd(currentDir);

filename = 'Plastic_data.xlsx';
data = xlsread(filename);
input = data(3:end,4:end);
output = data(1,4:end);
wavenumber = data(3:end,3);

figure;
plot(wavenumber,input); xlim([5900,8100]);
xlabel('Wavenumber(cm-1)'); ylabel('Reflectance');

run('Plastic\Modle_Plastic.m');
test_output=outputlog(:,1);
ture_output=outputlog(:,2);
figure;
h = confusionchart(ture_output, test_output);
xlabel('Predicted class'); ylabel('Actual class'); 
title({'Confusion Matrix';['Accuracy : ',num2str(accuracy.*100),'%']});