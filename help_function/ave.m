function [positive_ave,negative_ave,posi_pair_dist,negati_pair_dist] = ave(current_L,side_info,train_data)
% compute the average distance of all positive pairs and negative pairs
% with learned metric

positive_ave = 0;
negative_ave = 0;

positive = find(side_info(:,3)==1);
num_postive = size(positive,1);
posi_pair_dist = zeros(num_postive,1);
negative = find(side_info(:,3)==-1);
num_negative = size(negative,1);
negati_pair_dist = zeros(num_negative,1);

proj_data = train_data * current_L;

for i = 1:num_postive
    posi_pair_dist(i,1) = norm(proj_data(side_info(positive(i,1),1),:)-proj_data(side_info(positive(i,1),2),:));
%     posi_pair_dist(i,1) = (train_data(side_info(positive(i,1),1),:)-train_data(side_info(positive(i,1),2),:))*...
%         current_L*(train_data(side_info(positive(i,1),1),:)-train_data(side_info(positive(i,1),2),:))';
    positive_ave = positive_ave + posi_pair_dist(i,1);
end

for i = 1:num_negative
    negati_pair_dist(i,1) = norm(proj_data(side_info(negative(i,1),1),:)-proj_data(side_info(negative(i,1),2),:));
%     negati_pair_dist(i,1) = (train_data(side_info(negative(i,1),1),:)-train_data(side_info(negative(i,1),2),:))*...
%         current_L*(train_data(side_info(negative(i,1),1),:)-train_data(side_info(negative(i,1),2),:))';
    negative_ave = negative_ave + negati_pair_dist(i,1);
end

positive_ave = positive_ave/num_postive;
negative_ave = negative_ave/num_negative;