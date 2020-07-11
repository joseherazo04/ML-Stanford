function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));			%vector 2x1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% =========================================================================

%We need to add a new column of ones in X

h = X*theta;								%vector 12x1

%Cost without regularization
J = (1/(2*m))*sum((h-y).^2); 

%Cost with regularization
temptheta = theta;
temptheta(1,:) = 0; %discart first element from regularization

J = J + (lambda/(2*m))*sum(temptheta.^2);

%Gradients
I = eye(size(theta,1)); %using identity matrix to avoid changes in theta1
I(1,1) = 0; %to discart theta0 from regularization
grad = (1/m)*(X'*(h - y)) + (lambda/m)*I*theta;


grad = grad(:);

end
