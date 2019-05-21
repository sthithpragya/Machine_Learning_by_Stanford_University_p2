function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

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
X_temp = [ones(m,1) X];
z2 = X_temp*Theta1.';
a2 = sigmoid(z2);
a2_temp = [ones(size(a2,1),1) a2];
z3 = a2_temp*Theta2.';
a3 = sigmoid(z3);
h = a3;

% X = [ones(m,1) X];
% z2 = X*Theta1.';
% a2 = sigmoid(z2);
% a2 = [ones(size(a2,1),1) a2];
% z3 = a2*Theta2.';
% a3 = sigmoid(z3);
% h = a3;


K = max(y); %dimensionality of output
L = 3; % no. of layers

Y = cell(m,1);
for i = 1:m
    temp1 = zeros(K,1);
    temp1(y(i)) = 1;
    Y{i} = temp1;
end

%finding cost J only
for i = 1:m
    H = h(i,:).';
    y_val = Y{i};
    for k = 1:K
        J = J + y_val(k)*log(H(k)) + (1-y_val(k))*(log(1-H(k)));
    end
end

J = -J/m;

a = cell(L,1);
a_s = cell(L,1);

a{1} = X;
a{2} = a2;
a{3} = a3;

a_s{1} = X_temp;
a_s{2} = a2_temp;
a_s{3} = a3;


Theta = cell(L-1,1);
Theta{1} = Theta1;
Theta{2} = Theta2;

%finding regularisation term
R = 0;
% for l = 1:(L-1)
%     theta = Theta{l};
%     for p = 1:size(a_s{l},2)
%         for q = 1:size(a_s{l+1},2)
%             R = R + (theta(q,p))^2;
%         end
%     end
% end

for l = 1:L-1
    t = Theta{l}.^2;
    t = t(:,2:end);
    R = R + sum(t(:));
end

J
r = (lambda/(2*m))*R
J = J + r

%gradient calculation
D = cell(L,1); %D{1} and D{L} are defunct
tri = cell(L,1); %tri{1} and tri{L} are defunct
del = cell(L,1); %del{1} is defunct since a1 is X i.e. given

for i = 1:L
    D{i} = 0;
    tri{i} = 0;
    del{i} = 0;
end

for i = 1:m
    A_L = a_s{L};
    A_L = A_L(i,:).';
    del{L} = A_L - Y{i};
    for l = (L-1):-1:2
        A = a_s{l};
        A = A(i,:).';
        del{l} = (((Theta{l}).')*del{l+1}).*(A.*(1-A));
        del_temp = del{l};
        del_temp = del_temp(2:end);
        del{l} = del_temp;
    end
    for l = L-1:-1:1
        A = a_s{l};
        A = A(i,:).';
        tri{l} = tri{l} + del{l+1}*A.';
        d_temp = tri{l} + lambda*Theta{l};
        trii = tri{l};
        d_temp(:,1) = trii(:,1);
        D{l} = (1/m)*d_temp;
    end
end

Theta1_grad = D{1};
Theta2_grad = D{2};
% -------------------------------------------------------------
% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end