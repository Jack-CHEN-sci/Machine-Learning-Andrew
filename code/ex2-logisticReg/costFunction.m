function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Hypothesis function
h = sigmoid(X * theta);

% Cost function
J = (-y' * log(h) - (1-y)' * log(1-h)) / m;

% Gradient
grad = (X' * (h - y)) ./ m;

end
