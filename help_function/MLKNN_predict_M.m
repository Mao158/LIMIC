% function [Outputs, Pre_Labels] = MLKNN_predict_M(test_data, train_data, train_target, num_neighbour, Prior, PriorN, Cond, CondN, M)
% %MLKNN_predict uses trained multi-label k-nearest neighbor classifier to predict
% %
% %   Syntax
% %
% %       [Pred_label, Pred_prob] = MLKNN_predict_M(test_data, train_data, train_label, num_neighbour, Prior, PriorN, Cond, CondN, M)
% %
% %   Description
% %
% %       MLKNN_predict_M takes,
% %           test_data     - An num_test x num_dim array,the ith instance of testing instances is stored in test_data(i, :)
% %           train_data    - An num_train x num_dim array, the ith instance of training instances is stored in train_data(i, :)
% %           train_target  - An num_label x num_train array, if the ith instance of training instances belongs to the jth class, then train_labels(j, i) equals to +1, otherwise train_label(i, j) equals to 0
% %           num_neighbour - Number of neighbors used in the k-nearest neighbor algorithm
% %           Prior         - A num_label x 1 array, the prior probability P[Hj] representing the probability that an instance belongs to the jth class is stored in Prior(j, 1)
% %           PriorN        - A num_label x 1 array, the prior probability P[~Hj] representing the probability that an instance does not belong to the jth class is stored in PriorN(j, 1)
% %           Cond          - A num_label x (num_neighbour + 1) array, the likelihood P[k|Hj] representing the probability that k neighbors of an instance belonging to the jth class belong to the jth class is stored in Cond(j, k)
% %           CondN         - A num_label x (num_neighbour + 1) array, the likelihood P[k|~Hj] representing the probability that k neighbors of an instance not belonging to the jth class belong to the jth class is stored in CondN(j, k)
% %           M             - Distance metric
% %       and returns,
% %           Pre_Labels    - An num_label x num_test array, if the ith instance of testing instances belongs to the jth class, then Pred_label(j, i) equals to +1, otherwise Pred_label(j, i) equals to -1
% %           Outputs       - An num_label x num_test array, the probability of the ith instance of testing instances belonging to the jth class is stored in Pred_porb(j, i)
% 
% [num_test, ~] = size(test_data);
% [num_label, num_train] = size(train_target);
% 
% % Identifying k-nearest neighbors
% % computing distance between testing instances and training instances
% neighbours = zeros(num_test, num_neighbour);
% dist_matrix = zeros(1, num_train);
% for i = 1 : num_test
%     instance_i = test_data(i, :);
%     for j = 1 : num_train
%         instance_j = train_data(j, :);
%         dist_matrix(1, j) = (instance_i - instance_j) * M * (instance_i - instance_j)';
%     end
%     [~, sort_index] = sort(dist_matrix(1,:), 2);
%     neighbours(i,:) = sort_index(1, 1:num_neighbour);
% end
% 
% % Computing probability
% Outputs = zeros(num_label, num_test);
% prob_in = zeros(num_label, 1); % The probability P[Hj]*P[k|Hj] is stored in prob_in(j)
% prob_out = zeros(num_label, 1); % The probability P[~Hj]*P[k|~Hj] is stored in prob_out(j)
% for i = 1:num_test
%     temp_C = sum((train_target(:, neighbours(i, :))), 2); % The number of nearest neighbors belonging to the jth class is stored in temp_C(j, 1)
%     for j = 1:num_label
%         prob_in(j) = Prior(j) * Cond(j, temp_C(j) + 1);
%         prob_out(j) = PriorN(j) * CondN(j, temp_C(j) + 1);
%     end
%     Outputs(:, i) = prob_in ./ (prob_in + prob_out);
% end
% 
% % Assigning labels for testing instances
% Pre_Labels = ones(num_label, num_test);
% Pre_Labels(Outputs <= 0.5) = -1;

function [Outputs, Pre_Labels] = MLKNN_predict_M(test_data, train_data, train_target, num_neighbour, Prior, PriorN, Cond, CondN, M)

[num_test, ~]  = size(test_data);
[num_label, ~] = size(train_target);

% ===== Vectorized Mahalanobis distance =====
Xtr  = train_data;
Xte  = test_data;

XtrM = Xtr * M;
XteM = Xte * M;

xtrMxtr = sum(XtrM .* Xtr, 2)';   % 1 x N
xteMxte = sum(XteM .* Xte, 2);    % Nt x 1

dist_mat = xteMxte + xtrMxtr - 2 * (XteM * Xtr');

[~, neighbours] = mink(dist_mat, num_neighbour, 2);

% ===== Probability estimation =====
Outputs = zeros(num_label, num_test);

for i = 1:num_test
    temp_C = sum(train_target(:, neighbours(i,:)), 2); % num_label x 1
    idx = temp_C + 1;

    lin_idx = sub2ind(size(Cond), (1:num_label)', idx);

    prob_in  = Prior  .* Cond(lin_idx);
    prob_out = PriorN .* CondN(lin_idx);

    Outputs(:, i) = prob_in ./ (prob_in + prob_out);
end

% ===== Label decision =====
Pre_Labels = ones(num_label, num_test);
Pre_Labels(Outputs <= 0.5) = -1;
end
