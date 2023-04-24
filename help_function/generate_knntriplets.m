function [triplets, pairset] = generate_knntriplets(data, target, num_target_neighbour, num_imposter)

is_nearest = 1;
is_farthest = 0;

[num_data, ~] = size(data);

triplets = zeros(num_data * num_target_neighbour * num_imposter, 3);

class = unique(target);
diff_index = zeros(num_imposter, num_data);

for cc = 1:length(class)
%     fprintf('%i nearest imposter neighbors for class %i :\n',num_imposter,class(cc));
    i = find(target == class(cc));
    j = find(target ~= class(cc));
    class_size_imposter = size(j,1);
    nn = LSKnn(data(j,:)', data(i,:)', 1:num_imposter, class_size_imposter, is_nearest);
    
    % if class_size too small to find enough imposter neighbors
    % we keep the elements to zero and delete them later
    diff_index(1:min(num_imposter, class_size_imposter), i) = j(nn(1:min(num_imposter, class_size_imposter), :));
end

same_index = zeros(num_target_neighbour, num_data);
for cc = 1:length(class)
%     fprintf('%i farthest genuine neighbors for class %i:\n',num_target_neighbour,class(cc));
    i = find(target == class(cc));
    class_size = size(i,1);
%     nn=LSKnn(data(i,:)', data(i,:)', 1:num_target_neighbour, class_size, is_farthest);
    nn = LSKnn(data(i,:)',data(i,:)', 2:num_target_neighbour + 1, class_size, is_nearest);

    % if class_size too small to find enough genuine neighbors
    % we keep the elements to zero and delete them later
    same_index(1:min(num_target_neighbour, class_size - 1), i) = i(nn(1:min(num_target_neighbour, class_size - 1), :));
end

clear i j nn;
triplets(:, 1) = vec(repmat([1:num_data], num_target_neighbour * num_imposter, 1));
temp = zeros(num_target_neighbour * num_imposter, num_data);
for i = 1:num_target_neighbour
    temp((i - 1) * num_imposter + 1 : i * num_imposter, :) = repmat(same_index(i, :), num_imposter, 1);
end
triplets(:, 2) = vec(temp);

triplets(:, 3) = vec(repmat(diff_index, num_target_neighbour, 1));

pairset = zeros(2, num_data * num_target_neighbour);
pairset(1,:) = vec(repmat([1:num_data], num_target_neighbour, 1));
pairset(2,:) = vec(same_index);

% remove missing triplets / pairs (0 valued)
triplets = triplets(all(triplets, 2), :);
pairset = pairset(:, all(pairset, 1));

% fprintf('totally %d triplets for training\n\n', size(triplets,1));
