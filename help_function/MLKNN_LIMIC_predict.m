% function [Outputs, Pre_Labels] = MLKNN_LIMIC_predict(test_data, train_data, train_target, num_neighbour, Prior, PriorN, Cond, CondN, L, para)
% % MLKNN_LIMIC_predict uses trained multi-label k-nearest neighbor classifier to predict
% %
% %   Syntax
% %
% %       [Outputs, Pre_Labels] = MLKNN_LIMIC_predict(test_data, train_data, train_label, num_neighbour, Prior, PriorN, Cond, CondN, L, para)
% %
% %   Description
% %
% %       MLKNN_LIMIC_predict takes,
% %           test_data     - An num_test x num_dim array,the ith instance of testing instances is stored in test_data(i, :)
% %           train_data    - An num_train x num_dim array, the ith instance of training instances is stored in train_data(i, :)
% %           train_target  - An num_label x num_train array, if the ith instance of training instances belongs to the jth class, then train_labels(j, i) equals to +1, otherwise train_label(i, j) equals to 0
% %           num_neighbour - Number of neighbors used in the k-nearest neighbor algorithm
% %           Prior         - A num_label x 1 array, the prior probability P[Hj] representing the probability that an instance belongs to the jth class is stored in Prior(j, 1)
% %           PriorN        - A num_label x 1 array, the prior probability P[~Hj] representing the probability that an instance does not belong to the jth class is stored in PriorN(j, 1)
% %           Cond          - A num_label x (num_neighbour + 1) array, the likelihood P[k|Hj] representing the probability that k neighbors of an instance belonging to the jth class belong to the jth class is stored in Cond(j, k)
% %           CondN         - A num_label x (num_neighbour + 1) array, the likelihood P[k|~Hj] representing the probability that k neighbors of an instance not belonging to the jth class belong to the jth class is stored in CondN(j, k)
% %           L             - An num_label x (num_dim^2) array, Local Distance metrics
% %           para          - parameters in need
% %       and returns,
% %           Pre_Labels    - An num_label x num_test array, if the ith instance of testing instances belongs to the jth class, then Pred_label(j, i) equals to +1, otherwise Pred_label(j, i) equals to -1
% %           Outputs       - An num_label x num_test array, the probability of the ith instance of testing instances belonging to the jth class is stored in Pred_porb(j, i)
% 
% [num_test, ~] = size(test_data);
% [num_label, ~] = size(train_target);
% 
% % Identifying k-nearest neighbors under different metrics
% % computing distance between testing instances and training instances
% neighbours = cell(num_label, 1);
% for i = 1 : num_label
%     if para.with_global == true
%         % compute distance between each test_data and train_data with L_i+L_0
%         current_L = mat(L(i,:)' + L(end,:)');
%     else
%         % compute distance between each test_data and train_data with L_i
%         current_L = mat(L(i,:)');
%     end
% 
%     proj_train_data = train_data * current_L;
%     proj_test_data = test_data * current_L;
% 
%     [neighbour, ~] = knnsearch(proj_train_data, proj_test_data,'K', num_neighbour);
%     neighbours{i} = neighbour;
% end
% 
% % Computing probability
% Outputs = zeros(num_label, num_test);
% prob_in = zeros(num_label, 1); % The probability P[Hj]*P[k|Hj] is stored in prob_in(j)
% prob_out = zeros(num_label, 1); % The probability P[~Hj]*P[k|~Hj] is stored in prob_out(j)
% for i = 1:num_test
%     temp_C = zeros(num_label, 1);
%     for j = 1:num_label
%         temp_C(j) = sum((train_target(j, neighbours{j}(i, :))), 2); % The number of nearest neighbors belonging to the jth class is stored in temp_C(j, 1)
%         prob_in(j) = Prior(j) * Cond(j, temp_C(j) + 1);
%         prob_out(j) = PriorN(j) * CondN(j, temp_C(j) + 1);
%     end
%     Outputs(:, i) = prob_in ./ (prob_in + prob_out);
% end
% 
% % Assigning labels for testing instances
% Pre_Labels = ones(num_label, num_test);
% Pre_Labels(Outputs <= 0.5) = -1;

function [Outputs, Pre_Labels] = MLKNN_LIMIC_predict(test_data, train_data, train_target, num_neighbour, Prior, PriorN, Cond, CondN, L, para)
% MMMLC_MLKNN_predict predicts labels using a trained multi-label kNN classifier with per-label local metrics.
%
% Syntax:
%   [Outputs, Pre_Labels] = MMMLC_MLKNN_predict(test_data, train_data, train_target, num_neighbour, ...
%                                               Prior, PriorN, Cond, CondN, L, para)
%
% Inputs:
%   test_data     - num_test x d
%   train_data    - num_train x d
%   train_target  - num_label x num_train (binary: 0/1 or 0/+1)
%   num_neighbour - K (positive integer)
%   Prior         - num_label x 1
%   PriorN        - num_label x 1
%   Cond          - num_label x (K+1)
%   CondN         - num_label x (K+1)
%   L             - num_label x (d^2) (or (num_label+1) x d^2 if with_global)
%   para          - struct with field 'with_global' (true/false)
%
% Outputs:
%   Outputs       - num_label x num_test, posterior probability P(Cj | x)
%   Pre_Labels    - num_label x num_test, +1 (positive) or -1 (negative)

%% Input validation
assert(ndims(test_data) == 2 && ndims(train_data) == 2, 'Data must be 2D.');
[num_test, d_test] = size(test_data);
[num_train, d_train] = size(train_data);
assert(d_test == d_train, 'test_data and train_data must have same number of features.');

[num_label, num_train_target] = size(train_target);
assert(num_train == num_train_target, 'train_data and train_target instance counts mismatch.');

% Validate L
num_dim_sq = size(L, 2);
r = round(sqrt(num_dim_sq));
assert(r * r == num_dim_sq, 'L must have d^2 columns.');
num_dim = r;
assert(num_dim == d_test, 'Feature dimension mismatch between data and L.');
assert(size(L,1) == num_label || (para.with_global && size(L,1) == num_label + 1), ...
       'L row count must match num_label (or num_label+1 if with_global=true).');

% Validate model parameters
K = num_neighbour;
assert(K > 0 && floor(K) == K, 'num_neighbour must be positive integer.');
assert(all(size(Prior) == [num_label, 1]) && all(size(PriorN) == [num_label, 1]), 'Prior size mismatch.');
assert(all(size(Cond) == [num_label, K+1]) && all(size(CondN) == [num_label, K+1]), 'Cond/CondN size mismatch.');

%% Preallocate outputs
Outputs = zeros(num_label, num_test);
Pre_Labels = ones(num_label, num_test);

%% Main prediction loop over labels (can be parfor if needed)
for j = 1:num_label
    % Build local metric
    if para.with_global
        L_vec = L(j, :) + L(end, :);
    else
        L_vec = L(j, :);
    end
    L_mat = reshape(L_vec, num_dim, num_dim);  % d x d

    % Project data
    proj_train = train_data * L_mat;   % num_train x d
    proj_test  = test_data  * L_mat;   % num_test  x d

    % Find K nearest neighbors in training set for each test sample
    [idx, ~] = knnsearch(proj_train, proj_test, 'K', K);  % idx: num_test x K

    % Count how many of the K neighbors belong to label j
    neigh_labels = train_target(j, idx);        % 1 x (num_test*K) → auto to num_test x K
    neigh_labels = reshape(neigh_labels, size(idx));  % num_data x K
    k_counts = sum(neigh_labels, 2);            % num_test x 1

    % Compute posterior: P(Cj | k) ∝ P(Cj) * P(k | Cj)
    % Use k_counts + 1 as column index into Cond/CondN
    prob_in  = Prior(j)  .* Cond(j, k_counts + 1);   % num_test x 1
    prob_out = PriorN(j) .* CondN(j, k_counts + 1);  % num_test x 1

    % Avoid division by zero (though unlikely with smoothing)
    denom = prob_in + prob_out;
    denom(denom == 0) = eps;

    Outputs(j, :) = (prob_in ./ denom)';

    % Thresholding
    Pre_Labels(j, Outputs(j, :) <= 0.5) = -1;
end
end
