function [Outputs, Pre_Labels] = MLKNN_LIMIC_predict(test_data, train_data, train_target, num_neighbour, Prior, PriorN, Cond, CondN, L, para)
% MLKNN_LIMIC_predict uses trained multi-label k-nearest neighbor classifier to predict
%
%   Syntax
%
%       [Outputs, Pre_Labels] = MLKNN_LIMIC_predict(test_data, train_data, train_label, num_neighbour, Prior, PriorN, Cond, CondN, L, para)
%
%   Description
%
%       MLKNN_LIMIC_predict takes,
%           test_data     - An num_test x num_dim array,the ith instance of testing instances is stored in test_data(i, :)
%           train_data    - An num_train x num_dim array, the ith instance of training instances is stored in train_data(i, :)
%           train_target  - An num_label x num_train array, if the ith instance of training instances belongs to the jth class, then train_labels(j, i) equals to +1, otherwise train_label(i, j) equals to 0
%           num_neighbour - Number of neighbors used in the k-nearest neighbor algorithm
%           Prior         - A num_label x 1 array, the prior probability P[Hj] representing the probability that an instance belongs to the jth class is stored in Prior(j, 1)
%           PriorN        - A num_label x 1 array, the prior probability P[~Hj] representing the probability that an instance does not belong to the jth class is stored in PriorN(j, 1)
%           Cond          - A num_label x (num_neighbour + 1) array, the likelihood P[k|Hj] representing the probability that k neighbors of an instance belonging to the jth class belong to the jth class is stored in Cond(j, k)
%           CondN         - A num_label x (num_neighbour + 1) array, the likelihood P[k|~Hj] representing the probability that k neighbors of an instance not belonging to the jth class belong to the jth class is stored in CondN(j, k)
%           L             - An num_label x (num_dim^2) array, Local Distance metrics
%           para          - parameters in need
%       and returns,
%           Pre_Labels    - An num_label x num_test array, if the ith instance of testing instances belongs to the jth class, then Pred_label(j, i) equals to +1, otherwise Pred_label(j, i) equals to -1
%           Outputs       - An num_label x num_test array, the probability of the ith instance of testing instances belonging to the jth class is stored in Pred_porb(j, i)

[num_test, ~] = size(test_data);
[num_label, ~] = size(train_target);

% Identifying k-nearest neighbors under different metrics
% computing distance between testing instances and training instances
neighbours = cell(num_label, 1);
for i = 1 : num_label
    if para.with_global == true
        % compute distance between each test_data and train_data with L_i+L_0
        current_L = mat(L(i,:)' + L(end,:)');
    else
        % compute distance between each test_data and train_data with L_i
        current_L = mat(L(i,:)');
    end

    proj_train_data = train_data * current_L;
    proj_test_data = test_data * current_L;

    [neighbour, ~] = knnsearch(proj_train_data, proj_test_data,'K', num_neighbour);
    neighbours{i} = neighbour;
end

% Computing probability
Outputs = zeros(num_label, num_test);
prob_in = zeros(num_label, 1); % The probability P[Hj]*P[k|Hj] is stored in prob_in(j)
prob_out = zeros(num_label, 1); % The probability P[~Hj]*P[k|~Hj] is stored in prob_out(j)
for i = 1:num_test
    temp_C = zeros(num_label, 1);
    for j = 1:num_label
        temp_C(j) = sum((train_target(j, neighbours{j}(i, :))), 2); % The number of nearest neighbors belonging to the jth class is stored in temp_C(j, 1)
        prob_in(j) = Prior(j) * Cond(j, temp_C(j) + 1);
        prob_out(j) = PriorN(j) * CondN(j, temp_C(j) + 1);
    end
    Outputs(:, i) = prob_in ./ (prob_in + prob_out);
end

% Assigning labels for testing instances
Pre_Labels = ones(num_label, num_test);
Pre_Labels(Outputs <= 0.5) = -1;
