function [L, obj] = LIMIC_L(train_data, train_target, para)
% Label-Specific Multi-semantics Metric Learning for Multi-Label Data 
% (transformation L)

with_global = para.with_global;
num_label = size(train_target,1); % number of label
num_target_neighbour = para.num_target_neighbour;
num_imposter = para.num_imposter;
[~, num_dim] = size(train_data); % dimension of data

% Compute adjacency matrix W for label-correlation exploitation
W = compute_label_correlation(train_target);

% Generate label-specific side information w.r.t each label
side_info = cell(num_label,1);
for i = 1:num_label
    [Tri, ~] = generate_knntriplets(train_data, train_target(i,:)', num_target_neighbour, num_imposter);
    must_link = unique(Tri(:,[1,2]), 'rows');% postive pairs
    cannot_link = unique(Tri(:,[1,3]), 'rows');% negative pairs
    clear Tri;
    side_info{i,1} = [[must_link, ones(size(must_link, 1), 1)]; [cannot_link, -ones(size(cannot_link, 1), 1)]];
    clear must_link cannot_link;
end

% initialize local and global metric
% each metric is stored in the manner of row vector
if with_global == true
    num_metric = num_label + 1; % number of metric(with a global metric)
    L = repmat(vec(zeros(num_dim))', num_metric - 1, 1);% initialize local metric with zero matrix
    L = [L; vec(eye(num_dim))'];% initialize global metric with idenity matrix
else
    num_metric = num_label; % number of metric(without a global metric)
    L = repmat(vec(eye(num_dim))', num_metric, 1);% initialize local metric with idenity matrix
end

L = L(:);

% Accelerated projected gradient descent optimization
[L, obj] = ACC_GD(L, W, train_data, side_info, para);

L = reshape(L, num_metric, []);
assert(size(L, 2) == num_dim^2);

% % compute the average distance of all positive pairs and negative pairs with learned metric
% positive_ave = zeros(num_label,1);
% negative_ave = zeros(num_label,1);
% positive_pair_dist = cell(num_label,1);
% negative_pair_dist = cell(num_label,1);
% for kk = 1:num_label
%     if para.with_global == true
%         current_L = mat(L(kk,:)') + mat(L(end,:)');
%     else
%         current_L = mat(L(kk,:)');
%     end
%     current_M = current_L*current_L';
%     [positive_ave(kk,1),negative_ave(kk,1),positive_pair_dist{kk,1},negative_pair_dist{kk,1}] = ave(current_M,side_info{kk},train_data);
% end


