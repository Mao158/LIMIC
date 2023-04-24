function [L,obj_val] = ACC_GD(L, W, train_data, side_info, para)
% Accelerated projected gradient descent optimization

lambda_1 = para.lambda_1;
lambda_2 = para.lambda_2;

verbose = para.verbose;
max_iter = para.max_iter;
with_global = para.with_global;
learn_rate = para.learn_rate;

obj_val = zeros(1, max_iter);
[obj, grad] = compute_grad_L(L, W, train_data, side_info, lambda_1, lambda_2, with_global);

if verbose
    fprintf('Iter | Objective | Objective difference | Learning rate\n');
end

for iter = 1:max_iter
    while true
        L_next = L - learn_rate .* grad;
        [obj_next, grad_next] = compute_grad_L(L_next, W, train_data, side_info, lambda_1, lambda_2, with_global);
        delta_obj = obj_next - obj;

        if delta_obj > 0
            learn_rate = learn_rate / 2;
        else
            break;
        end
    end
    L = L_next;
    grad = grad_next;
    obj = obj_next;
    learn_rate = learn_rate * 1.01;

    obj_val(iter) = obj;

    if verbose
        fprintf('%4d | %9.5f | %15.5f      | %10f\n', iter, obj, delta_obj, learn_rate);
    end

    if iter > 50 && abs(delta_obj) < 0.001
        if verbose
            fprintf('LIMIC converged at %d-th iteration with objective %f\n', iter, obj);
        end
        break;
    elseif iter == max_iter && abs(delta_obj) >= 0.001
        if verbose
            fprintf('LIMIC did not converge in %d steps\n', iter);
        end
    end

end


end

