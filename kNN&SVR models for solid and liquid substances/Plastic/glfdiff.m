function dy = glfdiff(y,t,gam)
 
if strcmp(class(y),'function_handel')
    y = y(t);
end
 
h = t(2)-t(1);
w = 1;
y = y(:);
t = t(:);
 
for j = 2:length(t)
    w(j) = w(j-1)*(1-(gam+1)/(j-1));
end
for i = 1:length(t)
    dy(i) = w(1:i)*[y(i:-1:1)]/h^gam;
end
