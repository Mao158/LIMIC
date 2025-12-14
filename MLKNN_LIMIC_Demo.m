clc;
clear;

addpath(genpath('dataset'));
addpath(genpath('evaluation'));
addpath(genpath('help_function'));

% load('CAL500.mat');% 502,68,174
load('emotions.mat');% 593,72,6

% data = PCA(data);
% data = zscore(data);
[num_data, num_dim] = size(data);
num_label = size(target,1);
num_run = 10; % number of run
para.num_target_neighbour = 10;% number of target neighbours
para.num_imposter = 10;% number of imposters
para.learn_rate = 0.1;
para.max_iter = 200;
para.verbose = true;
para.with_global = true;

% Here, lambda_1 and lambda_2 should be tuned by model selection stratgies, such as 5-fold cross validation
para.lambda_1 = 1;
para.lambda_2 = 200;

% Parameters of MLKNN
num_MLKNN_neighbour = 10;
smooth = 1.0;

Result_MLKNN = zeros(num_run, 6);
Result_MLKNN_LIMIC = zeros(num_run, 6);

Result_Metric = cell(num_run, 1);
Iter_obj = zeros(num_run, para.max_iter);
Time = zeros(num_run, 1);

% Set a random seed to make the experiment reproducible
seed = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(seed);
indices = crossvalind('Kfold',num_data,10);

% parfor r = 1 : num_run
for r = 1 : num_run
    fprintf('*****Performing %d / %d -th run time*****\n',r,num_run);

    test_logical = (indices == r);
    train_logical = ~test_logical;
    train_data = data(train_logical,:);
    test_data = data(test_logical,:);
    train_target = target(:,train_logical);
    test_target = target(:,test_logical);

    % When encountering severe class-imbalance problem,
    % we ignore the corresponding label.
    num_train = size(train_data,1);
    sum_class = sum(train_target,2);
    logical_label = true(num_label,1);
    for kk = 1:num_label
        if sum_class(kk,1) <= 1 || ((num_train - sum_class(kk,1)) <= 1)
            logical_label(kk,1) = false;
        end
    end
    train_target = train_target(logical_label,:);
    test_target = test_target(logical_label,:);

    % Compute label-specific multi-semantics metrics for multi-label data
    tic;
    [L, obj] = LIMIC_L(train_data, train_target, para);

    % Saving training time
    Time(r) = toc;
    fprintf('The training time of %d-th run is %f\n', r, Time(r));   

    % Saving iteration objectives
    Iter_obj(r,:) = obj;

    % saving learned global and local metrics
    Result_Metric{r} = L;

    % Distance/similarity-based multi-label classification strategy: MLKNN
    [Prior, PriorN, Cond, CondN] = MLKNN_train_M(train_data, train_target, num_MLKNN_neighbour, smooth, eye(num_dim));
    [Outputs, Pre_Labels] = MLKNN_predict_M(test_data, train_data, train_target, num_MLKNN_neighbour, Prior, PriorN, Cond, CondN, eye(num_dim));
    [HammingLoss, RankingLoss, Coverage, Average_Precision, MacroF1, MacroAUC] = MLEvaluate(Outputs, Pre_Labels, test_target);
    res_MLKNN = [HammingLoss, RankingLoss, Coverage, Average_Precision, MacroF1, MacroAUC];
    Result_MLKNN(r,:) = res_MLKNN;

    % MLKNN is coupled with LIMIC: MLKNN-LIMIC
    [Prior_MLKNN_LIMIC, PriorN_MLKNN_LIMIC, Cond_MLKNN_LIMIC, CondN_MLKNN_LIMIC] = MLKNN_LIMIC_train(train_data, train_target, num_MLKNN_neighbour, smooth, L, para);
    [Outputs_MLKNN_LIMIC, Pre_Labels_MLKNN_LIMIC] = MLKNN_LIMIC_predict(test_data, train_data, train_target, num_MLKNN_neighbour, Prior_MLKNN_LIMIC, PriorN_MLKNN_LIMIC, Cond_MLKNN_LIMIC, CondN_MLKNN_LIMIC, L, para);
    [HammingLoss_MLKNN_LIMIC, RankingLoss_MLKNN_LIMIC, Coverage_MLKNN_LIMIC, Average_Precision_MLKNN_LIMIC, MacroF1_MLKNN_LIMIC, MacroAUC_MLKNN_LIMIC] = MLEvaluate(Outputs_MLKNN_LIMIC, Pre_Labels_MLKNN_LIMIC, test_target);
    res_MLKNN_LIMIC = [HammingLoss_MLKNN_LIMIC, RankingLoss_MLKNN_LIMIC, Coverage_MLKNN_LIMIC, Average_Precision_MLKNN_LIMIC, MacroF1_MLKNN_LIMIC, MacroAUC_MLKNN_LIMIC];
    Result_MLKNN_LIMIC(r,:) = res_MLKNN_LIMIC;
end
Result_MLKNN_mean = mean(Result_MLKNN,1);
Result_MLKNN_std = std(Result_MLKNN,0,1);
Result_MLKNN_LIMIC_mean = mean(Result_MLKNN_LIMIC,1);
Result_MLKNN_LIMIC_std = std(Result_MLKNN_LIMIC,0,1);

% Print results of MLKNN
fprintf('MLKNN results:\n');
fprintf(' %12s  %12s  %12s  %8s %12s  %12s\n','HammingLoss↓', 'RankingLoss↓', 'Coverage↓','Average_Precision↑', 'MacroF1↑', 'MacroAUC↑');
fprintf('%6.3f±%5.3f  %6.3f±%5.3f  %6.3f±%6.3f   %6.3f±%5.3f      %6.3f±%5.3f  %6.3f±%5.3f\n',Result_MLKNN_mean(1), Result_MLKNN_std(1), Result_MLKNN_mean(2), Result_MLKNN_std(2), ...
    Result_MLKNN_mean(3), Result_MLKNN_std(3), Result_MLKNN_mean(4), Result_MLKNN_std(4), Result_MLKNN_mean(5), Result_MLKNN_std(5), Result_MLKNN_mean(6), Result_MLKNN_std(6));

% Print results of MLKNN-LIMIC
fprintf('MLKNN-LIMIC results:\n');
fprintf(' %12s  %12s  %12s  %8s %12s  %12s\n','HammingLoss↓', 'RankingLoss↓', 'Coverage↓','Average_Precision↑', 'MacroF1↑', 'MacroAUC↑');
fprintf('%6.3f±%5.3f  %6.3f±%5.3f  %6.3f±%6.3f   %6.3f±%5.3f      %6.3f±%5.3f  %6.3f±%5.3f\n',Result_MLKNN_LIMIC_mean(1), Result_MLKNN_LIMIC_std(1), Result_MLKNN_LIMIC_mean(2), Result_MLKNN_LIMIC_std(2), ...
    Result_MLKNN_LIMIC_mean(3), Result_MLKNN_LIMIC_std(3), Result_MLKNN_LIMIC_mean(4), Result_MLKNN_LIMIC_std(4), Result_MLKNN_LIMIC_mean(5), Result_MLKNN_LIMIC_std(5), Result_MLKNN_LIMIC_mean(6), Result_MLKNN_LIMIC_std(6));


