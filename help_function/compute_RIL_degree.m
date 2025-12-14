function RIL = compute_RIL_degree(train_data, train_target, num_neighbour, rho, L, para)

[num_label, num_data] = size(train_target);

% Identifying k-nearest neighbors under different metrics
% computing distance between testing instances and training instances
neighbours = cell(num_label, 1);
for i = 1 : num_label
    if para.with_global == true
        % compute distance between each test_data and train_data with L_i+L_0
        current_L = mat(L(i,:)' + L(end,:)');
    else
        % compute distance between each test_data and train_data with L_i
        current_L = mat(L(i,:)');
    end

    proj_train_data = train_data * current_L;

    [neighbour_temp, ~] = knnsearch(proj_train_data, proj_train_data,'K', num_neighbour+1);
    neighbour = neighbour_temp(:,2:end);
    neighbours{i} = neighbour;
end


RIL = cell(num_data,1);

for index_data = 1:num_data
    g = zeros(num_label+1,1);
    for index_label = 1:num_label
        num_positive = 0;
        for index_nei = 1:num_neighbour
            nei = neighbours{index_label}(index_data,index_nei);
            if train_target(index_label,nei) == 1
                num_positive = num_positive + 1;
            end
        end
        g(index_label+1) = num_positive;
    end
    g(1) = (max(g(2:end))+min(g(2:end)))/2;
    
    sum_g = sum(g);
    for index = 1:num_label+1
        g(index) = g(index)/sum_g;
    end
    RIL{index_data} = g;
end






