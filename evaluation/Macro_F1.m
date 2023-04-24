function MacroF1 = Macro_F1(Pre_Labels,test_target)
% Computing the Macro F1
% Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
% test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1
Pre_Labels(Pre_Labels<0) = 0;
[num_class,~] = size(Pre_Labels);

MacroF1=0;

for j = 1:num_class
    temp = Pre_Labels(j,:).*test_target(j,:);
    TP = sum(temp);
    FP = sum(Pre_Labels(j,:))-sum(temp);
    FN = sum(test_target(j,:))-sum(temp);
    if 2*TP+FN+FP == 0
        F1 = 0;
    else
        F1 = 2*TP/(2*TP+FN+FP);
    end
    MacroF1 = MacroF1+F1;
end
MacroF1 = MacroF1/num_class;
end

