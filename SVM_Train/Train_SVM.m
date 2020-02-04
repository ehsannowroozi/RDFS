
clc
clear
close all

tic

% Add LIBSVM and NPY package
addpath(genpath('./Libsvm-3.17'));
addpath(genpath('./npy-matlab-master')) 

for jj=0:49
% load the features already computed
data_H0_matrix=sprintf('StammNet_subset_pristine_%d.npy',jj);
data_H1_matrix=sprintf('StammNet_subset_median08_%d.npy',jj);
data_H0_matrix = readNPY(data_H0_matrix);
data_H1_matrix = readNPY(data_H1_matrix);

%----------------------------------------------------------------------
% 1. Define the setup and prepare data
%----------------------------------------------------------------------

Ntest_H0 = 4000;
Ntest_H1 = Ntest_H0;


cv_idx_H0 = 500001:501000;    % 1000 for cross validation
cv_idx_H1 = cv_idx_H0;
tr_idx_H0 = 1:20000;          % 20000 for training
tr_idx_H1 = tr_idx_H0;
te_idx_H0 = 505001:509000;    % 4000 for testing
te_idx_H1 = te_idx_H0;


%----------------------------------------------------------------------
% 1. N-fold cross-validation
%----------------------------------------------------------------------

cross_labels = [zeros(1,numel(cv_idx_H0)), ones(1,numel(cv_idx_H1))]';

% Examples for cross-validation
cross_data = [data_H0_matrix(cv_idx_H0,:); data_H1_matrix(cv_idx_H1,:)];

% Grid of parameters
folds = 5;
[C,gamma] = meshgrid(-5:2:15, -15:2:3);

% Grid search
cv_acc = zeros(numel(C),1);
for i=1:numel(C)
    cv_acc(i) = svmtrain(cross_labels, cross_data, ...
        sprintf('-q -c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
end

% Pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

C;
gamma;

%----------------------------------------------------------------------
% 2. Training model using best_C and best_gamma
%----------------------------------------------------------------------

bestc = 2^C(idx);
bestg = 2^gamma(idx);

% Training on images from each class not used in cross-validation
trainLabel = [zeros(1,numel(tr_idx_H0)), ones(1,numel(tr_idx_H1))]';
trainData = [data_H0_matrix(tr_idx_H0,:); data_H1_matrix(tr_idx_H1,:)];

% Train (probabilistic model)
model_2C = svmtrain(trainLabel, trainData, ['-c ' num2str(bestc) ' -g ' num2str(bestg) ' -b 0']);

%--------------------------------------------------------------------------
%                                     Testing
%--------------------------------------------------------------------------
testLabel = [zeros(1,Ntest_H0), ones(1,Ntest_H1)]';
testData = [data_H0_matrix(te_idx_H0,:); data_H1_matrix(te_idx_H1,:)];

% Test
[predict_label, accuracy, decision_function] = svmpredict(testLabel, testData, model_2C, ' -b 0');

% % % %---------------------------------------------------------------------
% % % %                       Save The Results
% % % %---------------------------------------------------------------------
% % % 
results=sprintf('results_%d.mat',jj);
save([results],'model_2C','accuracy')

clear all


t=toc;
fprintf('Elapsed Time: %.3f sec.\n',t);
end

% % % %---------------------------------------------------------------------
% % % %                       Average Accuracy Train
% % % %---------------------------------------------------------------------
for jj=0:49
    
%acc0=load('results_0.mat','accuracy');
%acc0=acc0.accuracy(1);

acc=sprintf('results_%d.mat',jj);
acc=load(acc,'accuracy');
accc(jj+1)=acc.accuracy(1);

end

SUM=sum(accc);
avg=SUM/length(accc);

fprintf('Average Accuracy No Attack: %.3f \n',avg);
