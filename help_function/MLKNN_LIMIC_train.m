function [Prior, PriorN, Cond, CondN] = MLKNN_LIMIC_train(train_data, train_target, num_neighbour, smooth, L, para)
% MLKNN_LIMIC_train trains a multi-label k-nearest neighbor classifier
%
%    Syntax
%
%       [Prior, PriorN, Cond, CondN] = MLKNN_LIMIC_train(train_data, train_target, num_neighbour, smooth, L, para)
%
%    Description
%
%      MLKNN_LIMIC_train takes,
%           train_data    - An num_data x num_dim array, the ith instance of training instance is stored in train_data(i,:)
%           train_target  - A num_label x num_data array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals 0
%           num_neighbour - Number of neighbors used in the k-nearest neighbor algorithm
%           Smooth        - Smoothing parameter
%           L             - An num_label x (num_dim^2) array, Local Distance metrics
%           para          - parameters in need
%      and returns,
%           Prior         - A num_label x 1 array, for the ith label Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN        - A num_label x 1 array, for the ith label Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           Cond          - A num_label x (num_neighbour + 1) array, for the ith label Ci, the probability of P(k|Ci) (0<=k<=num_neighbour) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
%           CondN         - A num_label x (num_neighbour + 1) array, for the ith label Ci, the probability of P(k|~Ci) (0<=k<=num_neighbour) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)

[num_label, num_data] = size(train_target);

% Computing the prior probability
Prior = (sum(train_target, 2) + smooth) ./ (2 * smooth + num_data);
PriorN = 1 - Prior;


% Identifying k-nearest neighbors under different metrics
% computing distance between instances
neighbours = cell(num_label, 1);
for i = 1 : num_label
    if para.with_global == true
        % compute distance with L_i + L_0
        current_L = mat(L(i,:)' + L(end,:)');
    else
        % compute distance with L_i
        current_L = mat(L(i,:)');
    end

    proj_train_data = train_data * current_L;

    [K_neighbour_index, ~] = knnsearch(proj_train_data, proj_train_data,'K', num_neighbour + 1);
    neighbour = K_neighbour_index(:, 2:end); % delete itself which is located in the first column of matrix 'K_neighbour_index'
    neighbours{i} = neighbour;
end


% Computing the likelihood
Cond = zeros(num_label, num_neighbour + 1);
CondN = zeros(num_label, num_neighbour + 1);
for j = 1 : num_label
    temp_Cj = zeros(num_neighbour + 1, 1); % The number of instances belong to the jth label which has k nearest neighbors belonging to the jth label is stored in temp_Cj(k+1)
    temp_NCj = zeros(num_neighbour + 1, 1); % The number of instances does not belong to the jth class which has k nearest neighbors belonging to the jth class is stored in temp_NCj(k+1)

    for i = 1 : num_data
        temp_k = sum(train_target(j, neighbours{j}(i, :))); % temp_k nearest neightbors of the ith instance belong to the jth class
        if (train_target(j, i) == 1)
            temp_Cj(temp_k + 1) = temp_Cj(temp_k + 1) + 1;
        else
            temp_NCj(temp_k + 1) = temp_NCj(temp_k + 1) + 1;
        end
    end

    sum_Cj = sum(temp_Cj);
    sum_NCj = sum(temp_NCj);
    for k = 1 : (num_neighbour + 1)
        Cond(j, k) = (smooth + temp_Cj(k)) / ((num_neighbour + 1) * smooth + sum_Cj);
        CondN(j, k) = (smooth + temp_NCj(k)) / ((num_neighbour + 1) * smooth + sum_NCj);
    end
end



