function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Initialization
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% Note that X is a matrix where each column is a feature and each row 
% is an example. You need to perform the normalization separately
% for each feature. 

mu = mean(X);    %  calculate mean value for each row
sigma = std(X);  %  calculate standard deviation for each row
X_norm  = (X - repmat(mu,size(X,1),1)) ./  repmat(sigma,size(X,1),1);   % Mean Normalization

end
