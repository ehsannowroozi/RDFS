 
clear
close all
clc

addpath(genpath('./Libsvm-3.17'));
addpath(genpath('./npy-matlab-master'))

for jj=0:9
    
%acc0=load('results_0.mat','accuracy');
%acc0=acc0.accuracy(1);

acc=sprintf('results_%d.mat',jj);
acc=load(acc,'accuracy');
accc(jj+1)=acc.accuracy(1);

end

SUM=sum(accc);
avg=SUM/length(accc);








