function [Outputs, Pre_Labels] = BRKNN_LIMIC_predict(train_data, train_target, test_data, L, para)
%BRKNN_predict uses trained multi-label k-nearest neighbor classifier to predict
%
%   Syntax
%
%       [Outputs, Pre_Labels] = BRKNN_LIMIC_predict(train_data, train_target, test_data, L, para)
%
%   Description
%
%       BRKNN_LIMIC_predict takes,
%           train_data    - An num_train x num_dim array, the ith instance of training instances is stored in train_data(i, :)
%           train_target  - An num_label x num_train array, if the ith instance of training instances belongs to the jth class, then train_labels(j, i) equals to +1, otherwise train_label(i, j) equals to 0
%           test_data     - An num_test x num_dim array,the ith instance of testing instances is stored in test_data(i, :)
%           L             - An num_label x (num_dim^2) array, Local Distance metrics
%           para          - parameters in need
%       and returns,
%           Pre_Labels    - An num_label x num_test array, if the ith instance of testing instances belongs to the jth class, then Pred_label(j, i) equals to +1, otherwise Pred_label(j, i) equals to -1
%           Outputs       - An num_label x num_test array, the probability of the ith instance of testing instances belonging to the jth class is stored in Pred_porb(j, i)

num_test = size(test_data,1);
num_label = size(train_target,1);
K = 10;% parameter of k-NN
Outputs = zeros(num_label, num_test); % numerical-results([0,1]) of labels for each test instance
Pre_Labels = zeros(num_label, num_test); % logical-results({-1,1}) of labels for each test instance

for i = 1:num_label
    if para.with_global == true
        % compute distance between each test_data and train_data with L_i+L_0
        current_L = mat(L(i,:)' + L(end,:)');  
    else
        % compute distance between each test_data and train_data with L_i
        current_L = mat(L(i,:)');
    end

    proj_train_data = train_data * current_L;
    proj_test_data = test_data * current_L;

    % ascertain K-neighbours of each test data
    [K_neighbour_index,K_neighbour_dist] = knnsearch(proj_train_data,proj_test_data,'K',K);
    K_neighbour_dist = K_neighbour_dist .* K_neighbour_dist;

    K_neighbour_target_temp = train_target(i,K_neighbour_index);
    K_neighbour_target = reshape(K_neighbour_target_temp,[],K);
    % compute weight based on K-neighbour-distance
    % in the following way, the situation of K_neighbour_dist == 0 can be handled carefully 
    K_neighbour_dist_row = sum(K_neighbour_dist,2);
    Similarity_temp = bsxfun(@rdivide,K_neighbour_dist,K_neighbour_dist_row);
    Similarity_temp(isnan(Similarity_temp)) = 0.5;
    Similarity = ones(num_test,K) - Similarity_temp;
    sum_Similarity = sum(Similarity,2);
    Weight = bsxfun(@rdivide,Similarity,sum_Similarity);
    Outputs(i,:) = sum(Weight .* K_neighbour_target,2)';

    Pre_Labels(i,:) = Outputs(i,:);
    Pre_Labels(i,(Pre_Labels(i,:)>=0.5)) = 1;
    Pre_Labels(i,(Pre_Labels(i,:)<0.5)) = -1;
end