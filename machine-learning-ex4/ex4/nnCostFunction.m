function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% =========================================================================

%Note: I did this test sequentaly, that's way there are some reduntant steps here. But it doesn't matter for me.
%		Do you understand José of the future? I don't want to clean this code, I don't want now!

%========================PART 1===============================
% Transform y to vectors
y_vector = zeros(m, num_labels);

for i = 1:m
	y_vector(i, y(i)) = 1;
end

a1 = [ones(m, 1) X];

z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];

z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;

%without regularization
cost = zeros(m,1);

for i = 1:m
	% Little bit hard to make this operation work, specially because of the sizes of the vectors
	cost(i) = - y_vector(i,:)*log( h(i,:) )' - (1 - y_vector(i,:))*log( 1 - h(i,:) )';
	
	% This way also work
	%for j = 1:num_labels
	%	cost(i) = cost(i) + ( - y_vector(i,j) * log( h(i,j) ) - ( 1 - y_vector(i,j) )*log( 1 - h(i,j)));
	%end

end

% Non-regularized cost function
J = (1/m)*sum(cost);

%=======Regularization=======
tempTheta1 = Theta1;
tempTheta2 = Theta2;

%first element to zero
tempTheta1(:,1) = 0;
tempTheta2(:,1) = 0;

%sum them all
tempTheta1 = sum( sum( tempTheta1.^2 ));
tempTheta2 = sum( sum( tempTheta2.^2 ));
 
J = J + (lambda/(2*m))*(tempTheta1 + tempTheta2);


%========================PART 2===============================
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for i = 1:m
														%  X 	matrix 	5000x400
														%Theta1 matrix  25x401
														%Theta2 matrix 	10x26
	%Forward propagation
	a1 = X( i, :);   									%vector 1x400
	a1 = a1(:);      									%vector 400x1	Note: there is something reduntant here
	a1 = [1 ; a1];   									%vector 401x1		  I'll fix it in the future (...Probably)
	a1 = a1';        									%vector 1x401

	z2 = a1*Theta1'; 									%vector 1x25
	a2 = sigmoid(z2);									%vector 1x25
	a2 = [ones(size(a2, 1), 1) a2];						%vector 1x26
	
	z3 = a2*Theta2';
	a3 = sigmoid(z3);									%vector 1x10
	
	%Back propagation
	delta3 = a3 - y_vector( i,:);						%vector 1x10
	
	%delta2
	tempTheta2 = Theta2(:,2:end);						%matrix 10x25
	sig_grad = sigmoidGradient(z2);						%vector 1x25
	
	delta2 = (delta3*tempTheta2).*sig_grad;				%vector 1x25
	
	Delta2 = Delta2 + delta3' * a2;						%matrix 10x26
	Delta1 = Delta1 + delta2' * a1;						%matrix 25x401
	
end

Theta1_grad = (1/m).*Delta1;
Theta2_grad = (1/m).*Delta2;

%========================PART 3===============================
%Adding regularization
tempTheta1 = Theta1;
tempTheta2 = Theta2;

%first element to zero
tempTheta1(:,1) = 0;
tempTheta2(:,1) = 0;

Theta1_grad = Theta1_grad + (lambda/m).*tempTheta1;
Theta2_grad = Theta2_grad + (lambda/m).*tempTheta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end