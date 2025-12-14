% function [Prior,PriorN,Cond,CondN] = MLKNN_train_M(train_data, train_target, num_neighbour, smooth, M)
% %MLKNN_train trains a multi-label k-nearest neighbor classifier
% %
% %    Syntax
% %
% %       [Prior,PriorN,Cond,CondN] = MLKNN_train_M(train_data, train_target, num_neighbour, Smooth, M)
% %
% %    Description
% %
% %      MLKNN_train_M takes,
% %           train_data    - An num_data x num_dim array, the ith instance of training instance is stored in train_data(i,:)
% %           train_target  - A num_label x num_data array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals 0
% %           num_neighbour - Number of neighbors used in the k-nearest neighbor algorithm
% %           Smooth        - Smoothing parameter
% %           M             - Distance metric
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
% % Identifying k-nearest neighbors
% % computing distance between instances
% neighbours = zeros(num_data, num_neighbour);
% dist_matrix = zeros(1, num_data);
% for i = 1 : num_data
%     instance_i = train_data(i, :);
%     for j = 1 : num_data
%         if i == j
%             dist_matrix(1, j) = realmax; % setting the distance between an instance and itself to INF
%         else
%             instance_j = train_data(j, :);
%             dist_matrix(1, j) = (instance_i - instance_j) * M * (instance_i - instance_j)';
%         end
%     end
%     [~, sort_index] = sort(dist_matrix(1,:), 2);
%     neighbours(i,:) = sort_index(1, 1:num_neighbour);
% end
% 
% % Computing the likelihood
% Cond = zeros(num_label, num_neighbour + 1);
% CondN = zeros(num_label, num_neighbour + 1);
% for j = 1 : num_label
%     temp_Cj = zeros(num_neighbour + 1, 1); % The number of instances belong to the jth label which has k nearest neighbors belonging to the jth label is stored in temp_Cj(k+1)
%     temp_NCj = zeros(num_neighbour + 1, 1); % The number of instances does not belong to the jth class which has k nearest neighbors belonging to the jth class is stored in temp_NCj(k+1)
% 
%     for i = 1 : num_data
%         temp_k = sum(train_target(j, neighbours(i, :))); % temp_k nearest neightbors of the ith instance belong to the jth class
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

function [Prior,PriorN,Cond,CondN] = MLKNN_train_M(train_data, train_target, num_neighbour, smooth, M)

[num_label, num_data] = size(train_target);

% ===== Prior =====
Prior  = (sum(train_target, 2) + smooth) ./ (2*smooth + num_data);
PriorN = 1 - Prior;

% ===== Distance (vectorized Mahalanobis) =====
X  = train_data;
XM = X * M;
xMx = sum(XM .* X, 2);

dist_mat = xMx + xMx' - 2 * (XM * X');
dist_mat(1:num_data+1:end) = inf;

[~, neighbours] = mink(dist_mat, num_neighbour, 2);

% ===== Likelihood =====
Cond  = zeros(num_label, num_neighbour + 1);
CondN = zeros(num_label, num_neighbour + 1);

for j = 1:num_label
    label_j = train_target(j, :); % 1 × num_data
    k_count = sum(label_j(neighbours), 2); % k_count: num_data × 1

    pos = (label_j' == 1);
    neg = ~pos;

    temp_Cj  = accumarray(k_count(pos) + 1, 1, [num_neighbour+1,1]);
    temp_NCj = accumarray(k_count(neg) + 1, 1, [num_neighbour+1,1]);

    Cond(j,:)  = (smooth + temp_Cj')  / ((num_neighbour+1)*smooth + sum(temp_Cj));
    CondN(j,:) = (smooth + temp_NCj') / ((num_neighbour+1)*smooth + sum(temp_NCj));
end
end

