function [obj,grad] = compute_RELABKNN_grad(train_data, train_target, lambda, RIL,Theta)

[num_data, num_dim] = size(train_data);
num_label = size(train_target,1);

grad = zeros(num_dim,num_label+1);
obj = 0;

%Evaluate F accoding to Eq.(13)
F = zeros(num_data,num_label+1);
for i = 1:num_data
    for j = 1:num_label+1
        F(i,j) = exp(Theta(:,j)'*train_data(i,:)');
    end
end
%Normalization for F
Sum_F = sum(F,2);
for i = 1:num_data
    for j = 1:num_label+1
        F(i,j) = F(i,j)/Sum_F(i);
    end
end

%compute obj
obj_first_term = 0;
obj_second_term = 0;
for index_data = 1:num_data
    for index_label = 1:num_label+1
        %compute KL divergence
        obj_first_term = obj_first_term + sum(sum(RIL{index_data}'.* log(eps + RIL{index_data}'./(F(index_data,:)+eps))));

        postive_label_index = find(train_target(:,index_data) == 1);
        negative_label_index = find(train_target(:,index_data) == 0);
        r_i = size(postive_label_index,1)/size(negative_label_index,1);
        aug_postive_label_index = postive_label_index + 1; 
        aug_negative_label_index = negative_label_index + 1; 

        temp_positive = 0;
        temp_negative = 0;

        for index_positive = 1:size(aug_postive_label_index,1)
            temp_positive = temp_positive + (F(index_data,index_positive)-F(index_data,1));
        end

        for index_negative = 1:size(aug_negative_label_index,1)
            temp_negative = temp_negative + (F(index_data,1)-F(index_data,index_negative));
        end

        obj_second_term = obj_second_term - lambda*(temp_positive+r_i*temp_negative);
    end
    obj = obj + obj_first_term + obj_second_term;
end

%compute grad
grad_first_term = 0;
grad_second_term = 0;
for index_label = 1:num_label+1
    for index_data = 1:num_data
        % The first term
        grad_first_term = grad_first_term-(RIL{index_data}(index_label) - F(index_data,index_label))*train_data(index_data,:)';

        postive_label_index = find(train_target(:,index_data) == 1);
        negative_label_index = find(train_target(:,index_data) == 0);
        r_i = size(postive_label_index,1)/size(negative_label_index,1);
        aug_postive_label_index = postive_label_index + 1; 
        aug_negative_label_index = negative_label_index + 1; 

        %compute zeta
        if ismember(index_label,aug_postive_label_index)
            zeta = 1;
        elseif ismember(index_label,aug_negative_label_index)
            zeta = -r_i;
        else
            zeta = 0;
        end

        temp_positive = 0;
        temp_negative = 0;
        delete_aug_postive_label_index = aug_postive_label_index(~ismember(aug_postive_label_index,index_label));
        delete_aug_negative_label_index = aug_negative_label_index(~ismember(aug_negative_label_index,index_label));

        for index_delete_positive = 1:size(delete_aug_postive_label_index,1)
            temp_positive = temp_positive + (F(index_data,1)-F(index_data,index_delete_positive));
        end

        for index_delete_negative = 1:size(delete_aug_negative_label_index,1)
            temp_negative = temp_negative + (F(index_data,index_delete_negative)-F(index_data,1));
        end

        grad_second_term = grad_second_term-lambda*F(index_data,index_label)*(temp_positive+r_i*temp_negative+zeta*(1-F(index_data,index_label)+F(index_data,1)))*train_data(index_data,:)';
    end
    grad(:,index_label) = grad_first_term + grad_second_term;
end


end

