% function W = compute_label_correlation(train_target)
% % Compute adjacency matrix
% 
% % Compute conditional probability matrix P on training set
% num_label = size(train_target,1);
% P = zeros(num_label,num_label);
% for ii = 1:num_label
%     for jj = 1:num_label
%         if ii == jj
%             continue;
%         else
%             ii_idx = find(train_target(ii,:) == 1);
%             num_ii_idx = size(ii_idx,2);
%             jj_idx = find(train_target(jj,ii_idx) == 1);
%             num_jj_idx = size(jj_idx,2);
%             P(jj,ii) = num_jj_idx / num_ii_idx;
%         end
%     end
% end
% 
% % Compute adjacency matrix W based on P
% W = zeros(num_label,num_label);
% for ii = 1:num_label
%     for jj = ii+1:num_label
%             W(ii,jj) = 0.5*(P(jj,ii) + P(ii,jj));
%     end
% end
% W = W + W';
% end

function [W] = compute_label_correlation(train_target)
% Efficient computation of the label correlation matrix (no explicit for-loops)
% train_target: num_label x num_sample (binary matrix)

[num_label, ~] = size(train_target);

% Convert to double to avoid integer division
Y = double(train_target);

% Y * Y' gives the number of samples where each pair of labels are both 1
co_mat = Y * Y';  % num_label x num_label

% Number of samples where each label is 1 (column vector)
label_sum = sum(Y, 2);

% Avoid division by zero
label_sum(label_sum == 0) = eps;

% Conditional probability P(j,i) = co_mat(j,i) / label_sum(i)
P = co_mat ./ label_sum';   % Column-wise division (automatic broadcasting)

% Symmetrize to obtain W
W = 0.5 * (P + P');

% Remove diagonal elements (if self-correlation is not desired)
W(1:num_label+1:end) = 0;
end
