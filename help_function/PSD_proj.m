function output = PSD_proj(M)
% project multiple metrics onto PSD cone

size_M = size(M, 1);
% for each metric
output = zeros(size(M));
for mm = 1:size_M
    [VV, DD] = eig(symmetric(mat(M(mm,:)')));
    DD = diag(DD);
    DD(DD < 1e-10) = 0;
    output(mm,:) = vec(VV*diag(DD)*VV')';
end

    function output = symmetric(input)
        output = 0.5 * (input + input');
    end

end


