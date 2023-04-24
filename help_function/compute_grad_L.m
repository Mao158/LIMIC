function [obj,grad] = compute_grad_L(L, W, train_data, side_info, lambda_l, lambda_2, with_global)
% Compute gradient and objective value given current solution (transformation L)
[num_data,num_dim] = size(train_data);
num_label = size(side_info,1);% number of label

if with_global == true
    num_metric = size(side_info,1) + 1;% number with global metric
else
    num_metric = size(side_info,1);% number without global metric
end

% transform L from a vector to a matrix
L = reshape(L,num_metric,[]);
assert(size(L,2) == num_dim^2);
grad = zeros(size(L));
obj = 0;

%% for empirical loss term
gamma = 2;
if with_global == true
    for kk = 1:num_label
        num_T = size(side_info{kk},1);

        current_L = mat(L(kk,:)' + L(end,:)');
        proj_data = train_data * current_L;
        pair_distance = zeros(num_T,1);
        for jj = 1:num_T
            pair_distance(jj) = norm(proj_data(side_info{kk}(jj,1),:) - proj_data(side_info{kk}(jj,2),:))^2;
        end
        dist_vector = (gamma - pair_distance) .* side_info{kk}(:,end);

        % compute smooth hinge loss and delta from the dist_vector
        [obj_temp, delta] = compute_smooth_loss(dist_vector);
        obj = obj + obj_temp / num_T;

        %compute gradient for local and global metric
        delta_theta = delta .* side_info{kk}(:,end);
        SS = sparse(side_info{kk}(:,1), side_info{kk}(:,2), delta_theta, num_data, num_data);
        grad_temp = 2 * SODW(train_data', SS) * current_L;

        grad(kk, :) = grad(kk, :) + vec(grad_temp)'/num_T;
        grad(end, :) = grad(end, :) + vec(grad_temp)'/num_T;
    end
else
    for kk = 1:num_label
        num_T = size(side_info{kk},1);

        current_L = mat(L(kk,:)');
        proj_data = train_data * current_L;
        pair_distance = zeros(num_T,1);
        for jj = 1:num_T
            pair_distance(jj) = norm(proj_data(side_info{kk}(jj,1),:) - proj_data(side_info{kk}(jj,2),:))^2;
        end
        dist_vector = (gamma - pair_distance) .* side_info{kk}(:,end);

        % compute smooth hinge loss and delta from the dist_vector
        [obj_temp, delta] = compute_smooth_loss(dist_vector);
        obj = obj + obj_temp / num_T;

        %compute gradient for local and global metric
        delta_theta = delta .* side_info{kk}(:,end);
        SS = sparse(side_info{kk}(:,1), side_info{kk}(:,2), delta_theta, num_data, num_data);
        grad_temp = 2 * SODW(train_data', SS) * current_L;

        grad(kk, :) = grad(kk, :) + vec(grad_temp)'/num_T;
    end
end


%% for regularizers (F-norm on L)
for kk = 1:num_metric - 1
    obj = obj + lambda_l * norm(L(kk, :))^2;
    grad(kk, :) = grad(kk, :) + 2 * lambda_l * L(kk, :);
end

%% for label correlation term
obj_temp = 0;
for ii = 1:num_label
    grad_temp = zeros(1,size(grad,2));
    for jj = 1:num_label
        obj_temp = obj_temp + W(ii,jj) * norm(L(ii,:) - L(jj,:))^2;
        grad_temp = grad_temp + W(ii,jj) * (L(ii,:) - L(jj,:));
    end
    grad(ii,:) = grad(ii,:) + 4 * lambda_2 * grad_temp;
end
obj = obj + lambda_2 * obj_temp;

% transform gradient to a vector form
grad = grad(:);
end

