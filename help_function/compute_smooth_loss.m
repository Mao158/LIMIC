function [obj_temp, delta] = compute_smooth_loss(dist_vector)
% Compute smooth hinge loss and delta from dis_vector

num_T = size(dist_vector,1);
obj_temp = 0;
delta = zeros(num_T,1);

for ii = 1:num_T
    if dist_vector(ii) > 1
        obj_temp = obj_temp + 0;
        delta(ii,1) = 0;
    elseif dist_vector(ii) < 0
        obj_temp = obj_temp + 0.5 - dist_vector(ii);
        delta(ii,1) = 1;
    else
        obj_temp = obj_temp + 0.5 * (dist_vector(ii)-1) * (dist_vector(ii)-1);
        delta(ii,1) = 1 - dist_vector(ii);
    end

end

