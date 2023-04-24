function W = compute_label_correlation(train_target)
% Compute adjacency matrix

% Compute conditional probability matrix P on training set
num_label = size(train_target,1);
P = zeros(num_label,num_label);
for ii = 1:num_label
    for jj = 1:num_label
        if ii == jj
            continue;
        else
            ii_idx = find(train_target(ii,:) == 1);
            num_ii_idx = size(ii_idx,2);
            jj_idx = find(train_target(jj,ii_idx) == 1);
            num_jj_idx = size(jj_idx,2);
            P(jj,ii) = num_jj_idx / num_ii_idx;
        end
    end
end

% Compute adjacency matrix W based on P
W = zeros(num_label,num_label);
for ii = 1:num_label
    for jj = ii+1:num_label
            W(ii,jj) = 0.5*(P(jj,ii) + P(ii,jj));
    end
end
W = W + W';
end

