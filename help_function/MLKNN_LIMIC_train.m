% function [Prior, PriorN, Cond, CondN] = MLKNN_LIMIC_train(train_data, train_target, num_neighbour, smooth, L, para)
% % MLKNN_LIMIC_train trains a multi-label k-nearest neighbor classifier
% %
% %    Syntax
% %
% %       [Prior, PriorN, Cond, CondN] = MLKNN_LIMIC_train(train_data, train_target, num_neighbour, smooth, L, para)
% %
% %    Description
% %
% %      MLKNN_LIMIC_train takes,
% %           train_data    - An num_data x num_dim array, the ith instance of training instance is stored in train_data(i,:)
% %           train_target  - A num_label x num_data array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals 0
% %           num_neighbour - Number of neighbors used in the k-nearest neighbor algorithm
% %           Smooth        - Smoothing parameter
% %           L             - An num_label x (num_dim^2) array, Local Distance metrics
% %           para          - parameters in need
% %      and returns,
% %           Prior         - A num_label x 1 array, for the ith label Ci, the prior probability of P(Ci) is stored in Prior(i,1)
% %           PriorN        - A num_label x 1 array, for the ith label Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
% %           Cond          - A num_label x (num_neighbour + 1) array, for the ith label Ci, the probability of P(k|Ci) (0<=k<=num_neighbour) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
% %           CondN         - A num_label x (num_neighbour + 1) array, for the ith label Ci, the probability of P(k|~Ci) (0<=k<=num_neighbour) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)
% 
% [num_label, num_data] = size(train_target);
% 
% % Computing the prior probability
% Prior = (sum(train_target, 2) + smooth) ./ (2 * smooth + num_data);
% PriorN = 1 - Prior;
% 
% 
% % Identifying k-nearest neighbors under different metrics
% % computing distance between instances
% neighbours = cell(num_label, 1);
% for i = 1 : num_label
%     if para.with_global == true
%         % compute distance with L_i + L_0
%         current_L = mat(L(i,:)' + L(end,:)');
%     else
%         % compute distance with L_i
%         current_L = mat(L(i,:)');
%     end
% 
%     proj_train_data = train_data * current_L;
% 
%     [K_neighbour_index, ~] = knnsearch(proj_train_data, proj_train_data,'K', num_neighbour + 1);
%     neighbour = K_neighbour_index(:, 2:end); % delete itself which is located in the first column of matrix 'K_neighbour_index'
%     neighbours{i} = neighbour;
% end
% 
% 
% % Computing the likelihood
% Cond = zeros(num_label, num_neighbour + 1);
% CondN = zeros(num_label, num_neighbour + 1);
% for j = 1 : num_label
%     temp_Cj = zeros(num_neighbour + 1, 1); % The number of instances belong to the jth label which has k nearest neighbors belonging to the jth label is stored in temp_Cj(k+1)
%     temp_NCj = zeros(num_neighbour + 1, 1); % The number of instances does not belong to the jth class which has k nearest neighbors belonging to the jth class is stored in temp_NCj(k+1)
% 
%     for i = 1 : num_data
%         temp_k = sum(train_target(j, neighbours{j}(i, :))); % temp_k nearest neightbors of the ith instance belong to the jth class
%         if (train_target(j, i) == 1)
%             temp_Cj(temp_k + 1) = temp_Cj(temp_k + 1) + 1;
%         else
%             temp_NCj(temp_k + 1) = temp_NCj(temp_k + 1) + 1;
%         end
%     end
% 
%     sum_Cj = sum(temp_Cj);
%     sum_NCj = sum(temp_NCj);
%     for k = 1 : (num_neighbour + 1)
%         Cond(j, k) = (smooth + temp_Cj(k)) / ((num_neighbour + 1) * smooth + sum_Cj);
%         CondN(j, k) = (smooth + temp_NCj(k)) / ((num_neighbour + 1) * smooth + sum_NCj);
%     end
% end

function [Prior, PriorN, Cond, CondN] = MLKNN_LIMIC_train(train_data, train_target, num_neighbour, smooth, L, para)
% MMMLC_MLKNN_train trains a multi-label k-nearest neighbor classifier with per-label local metrics.
%
% Syntax:
%   [Prior,PriorN,Cond,CondN] = MMMLC_MLKNN_train(train_data, train_target, num_neighbour, smooth, L, para)
%
% Inputs:
%   train_data    - num_data x num_dim matrix
%   train_target  - num_label x num_data binary matrix (+1/0 or 1/0)
%   num_neighbour - number of nearest neighbors (K)
%   smooth        - smoothing parameter (>=0)
%   L             - num_label x (d^2) matrix, each row is vec(L_i) for label i
%   para          - struct with field 'with_global' (true/false)
%
% Outputs:
%   Prior         - num_label x 1, P(Ci)
%   PriorN        - num_label x 1, P(~Ci)
%   Cond(j,k+1)   - P(k neighbors in Ci | instance in Ci)
%   CondN(j,k+1)  - P(k neighbors in Ci | instance not in Ci)

%% Input validation
assert(ndims(train_data) == 2 && ndims(train_target) == 2, 'train_data and train_target must be 2D.');
[num_data, ~] = size(train_data);
[num_label, num_data_target] = size(train_target);
assert(num_data == num_data_target, 'Number of instances in train_data and train_target must match.');

assert(num_neighbour > 0 && floor(num_neighbour) == num_neighbour, 'num_neighbour must be a positive integer.');
assert(smooth >= 0, 'smooth must be non-negative.');

% Validate L dimension
num_dim_sq = size(L, 2);
assert(num_dim_sq > 0, 'L must have at least one column.');
r = round(sqrt(num_dim_sq));
assert(r * r == num_dim_sq, 'Number of columns in L must be a perfect square (d^2).');
num_dim = r;
assert(size(L, 1) == num_label || (para.with_global && size(L,1) == num_label + 1), ...
       'Number of rows in L must equal num_label (or num_label+1 if with_global=true).');

%% Compute prior probabilities
label_counts = sum(train_target, 2);  % num_label x 1
Prior = (label_counts + smooth) ./ (num_data + 2 * smooth);
PriorN = 1 - Prior;

%% Initialize output
K = num_neighbour;
Cond = zeros(num_label, K + 1);
CondN = zeros(num_label, K + 1);

%% Main loop over labels
for j = 1:num_label
    % Build local metric matrix L_j
    if para.with_global
        L_vec = L(j, :) + L(end, :);  % L(end,:) is global metric
    else
        L_vec = L(j, :);
    end
    L_mat = reshape(L_vec, num_dim, num_dim);  % d x d

    % Project data using local metric
    proj_X = train_data * L_mat;  % num_data x d

    % Find K+1 nearest neighbors (including self)
    [idx, ~] = knnsearch(proj_X, proj_X, 'K', K + 1);
    neigh_idx = idx(:, 2:end);  % num_data x K, exclude self

    % Count how many neighbors belong to label j
    neigh_labels = train_target(j, neigh_idx);      % 1 x (num_data*K) -> auto broadcast to num_data x K
    neigh_labels = reshape(neigh_labels, size(neigh_idx));  % num_data x K
    k_counts = sum(neigh_labels, 2);                % num_data x 1

    % Identify positive/negative instances for label j
    is_pos = logical(train_target(j, :)');          % num_data x 1

    % Vectorized counting using accumarray
    pos_k = k_counts(is_pos);
    neg_k = k_counts(~is_pos);

    temp_Cj = accumarray(pos_k + 1, 1, [K + 1, 1], @sum, 0);
    temp_NCj = accumarray(neg_k + 1, 1, [K + 1, 1], @sum, 0);

    % Apply Laplace smoothing
    Cond(j, :) = (temp_Cj' + smooth) ./ (sum(temp_Cj) + (K + 1) * smooth);
    CondN(j, :) = (temp_NCj' + smooth) ./ (sum(temp_NCj) + (K + 1) * smooth);
end
end




