%use mnist database next time
load("ex4data1.mat");
brightnessXaddeg
m = size(X, 1);
% Randomly select 4000 data points to train
sel = randperm(m);
Xtrain = X(sel(1:40000),:);
ytrain = y(sel(1:40000),:);
Xtest = X(sel(40001:45000),:);
ytest = y(sel(40001:45000),:);

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 100;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 500);
lambda = 0.3;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
 
pred = predict(Theta1, Theta2, Xtest);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predict(Theta1,Theta2,Xtrain) == ytrain)) * 100);