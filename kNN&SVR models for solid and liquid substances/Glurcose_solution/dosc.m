%#  function: [Z,R,P,T,newx] = dosc(Xcal,Ycal,Xtest,nocomp);
%#
%#  dosc Calculates orthogonal signal correction
%#  applying Direct Orthogonalization (Now Truly) 
%#
%#  The INPUTS are the matrix of predictor variables (Xcal)
%#  and predicted variable(s) (Ycal)and the matrix ofthe new 
%#  samples to predict (Xtest), scaled as desired, and the 
%#  number of OSC components to calculate (nocomp).
%#
%#  The OUTPUTS are the OSC corrected X-matrix (Z) and
%#  the weights (R), loads (P) and scores(T) that were
%#  used in making the correction and the new corrected(scaled) 
%#  x-data.
%#
%# Author: Sijmen de Jong, Oct 99
%#

function [Z,R,P,T,newx] = dosc(Xcal,Ycal,Xtest,nocomp)

pinvX = pinv(Xcal')';

% project Y onto X
Ycal = Xcal*(pinvX*Ycal);
% deflate X wrt (projected) Y
Z = Xcal-Ycal*(pinv(Ycal)*Xcal); 
% find major PCs 
OPTIONS.disp = 0;
[T,D] = eigs(Z*Z',nocomp,OPTIONS);
% deflate X wrt to these large-VAF, zero-correlation directions
Z = Xcal-T*(T'*Xcal); %
%Z = X - X*pinv(X)*T*(T'*X);
R = pinvX*T;
P =(T'*Xcal)';


newx = Xtest - Xtest*R*P';
