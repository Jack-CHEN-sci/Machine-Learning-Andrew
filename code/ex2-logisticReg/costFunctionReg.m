function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost of a particular choice of theta.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hypothesis function
h = sigmoid(X * theta);

% cost function
J = (-y' * log(h) - (1-y)' *  log(1-h)) ./ m + (lambda / (2 * m)) * (theta' * theta - theta(1)^2);  % ATTENTION: DO NOT regularize theta_0 !

% gradient
grad = X' * (h - y) ./ m + (lambda / m) * theta;
grad(1) = X(:,1)' * (h - y) / m;  % ATTENTION: the gradient of theta_0 need to be treated differently !

end
