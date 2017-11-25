function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X];

z_1 = X * Theta1';
a_1 = sigmoid(z_1);
a_1 = [ones(size(a_1,1), 1) a_1]; 
z_2 =  a_1 * Theta2'; 
a_2 = sigmoid(z_2); 
[~,p] = max(a_2, [], 2);
end
