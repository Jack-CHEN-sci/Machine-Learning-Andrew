function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

% Initialization
theta = zeros(size(X, 2), 1);

% Solution
theta = pinv(X'*X)*X'*y;

end
