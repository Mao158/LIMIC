function NN = LSKnn(X1, X2, ks, class_size, key)
B = 750;
[D, N] = size(X2);
NN = zeros(length(ks),N);

for i = 1:B:N
    BB = min(B,N-i);
    %   fprintf('.');
    Dist = distance(X1,X2(:,i:i+BB));
    %   fprintf('.');
    %   fprintf('.');
    %     if key == 1 % compute ks nearest points
    %         [dist,nn]=mink(Dist,max(ks));
    %     elseif key == 0 % compute ks farthest points
    %         [dist,nn]=maxk(Dist,max(ks));
    %     end
    if key == 1 % compute ks nearest points
        [~, idex] = sort(Dist,1,'ascend');
    elseif key == 0 % compute ks farthest points
        [~, idex] = sort(Dist,1,'descend');
    end
    clear('Dist');

    if class_size < max(ks)
        idex = [idex; zeros((max(ks)-class_size),size(idex,2))];
    end
    nn = idex(ks, :);
    NN(:,i:i+BB) = nn;
    clear('nn','dist');
    %   fprintf('(%i%%) ',round((i+BB)/N*100));
end

end

