function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

error = 10000; %this variable saves the slowest error (great number just to start)
sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30];
c_values = [0.01 0.03 0.1 0.3 1 3 10 30];

%loop over possible C
for c_value = c_values

	%loop over all posibles sigma
	for sigma_value = sigma_values
		
		fprintf('Training for c = %d and sigma = %d .\n', c_value, sigma_value);
		
		%train model
		model = svmTrain(X, y, c_value, @(x1, x2) gaussianKernel(x1, x2, sigma_value));
		
		%make predictions
		predictions = svmPredict(model, Xval);
		
		%evaluate error
		temp_error = mean(double(predictions ~= yval));
		
		%verify if the error is better then before
		if temp_error < error
			error = temp_error;
			C = c_value;
			sigma = sigma_value;
		end
	
	end
	
	fprintf('End up with c = %d and sigma = %d .\n', C, sigma);
end


% =========================================================================

end
