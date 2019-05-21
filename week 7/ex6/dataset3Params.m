function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

c = [0.01,0.03,0.1,0.3,1,3,10,30,100];
sig = [0.01,0.03,0.1,0.3,1,3,10,30,100];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
error = zeros(length(c),length(sig));
for i = 1:length(c)
    for j = 1:length(sig)
        model = svmTrain(X, y, c(i), @(x1, x2) gaussianKernel(x1, x2, sig(j))); 
        yval_predict = svmPredict(model,Xval);
        error(i,j) = sum(double(yval_predict ~= yval));
    end
end


error = error(:);
[~,I] = sort(error);
[m,n] = ind2sub([length(c),length(sig)],I(1));
C = c(m)
sigma = sig(n)
% =========================================================================
end
