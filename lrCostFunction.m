function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of parameters
h_x = sigmoid(X * theta);
J = sum((-y)' * log(h_x) - (ones(m,1)- y)' * log(ones(m,1)- h_x))/m; % Unregularized cost function.
J = J + (lambda/(2*m)) * (sum(theta.^2)-theta(1).^2);  % Adding regularizing factor.

grad = (h_x - y)' * X/m; % Unregularized gradient function.
grad(2:n) = grad(2:n) + ones(1,n-1).*theta(2:n)'*lambda/m; % Adding regularizing factor.
grad = grad';
end
