function proj_data = PCA(data)
%PCA: do PCA dimension reduction with 95% cumulative contribution preserved

[coeff,score,latent,tsquared,explained,mu] = pca(data);
cumulative_contribution = cumsum(latent)/sum(latent);
idx = find(cumulative_contribution > 0.95);
k = idx(1);
proj_data = score(:,1:k);

end

