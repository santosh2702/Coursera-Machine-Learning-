function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % first compute h_0(x) using current value of theta
    predictions = X*theta;

    % then compute delta
    % unvectorized version (remember that indices start from 1)
    delta(1) = 1/m * sum((predictions(1) - y(1))*X(1));
    % vectorized version
    delta = 1/m * sum(X' * (predictions - y));

    % finally compute updated theta
    % unvectorized version
    %theta(1) = 1/m * sum((predictions(1) - y(1))*x(1));
    %theta(2) = 1/m * sum((predictions(2) - y(2))*x(2));;
    % vectorized version
    theta = theta - alpha * delta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    % is this to plot it?

end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

