function accuracy = K_NN(trainDataProj,testDataProj,testLabel,K)
%
% This function implement the k-NN rule.
%
% accuracy = K_NN(trainDataProj,testDataProj,testLabel,K)
%
% trainDataProj - training data (each column is an observation)
% testDataProj - testing data (each column is an observation)
% testLabel - true label of testing data
% K - a designed parameter which decides the number of neighborhood
% accuracy - accuracy of the classifier (the value is within (0,1))
%
% Jing Xie
% 15/11/2020
%
    %Get the size of the data and initialize the parameters
    [~,N_Train] = size(trainDataProj);
    [~,N_Test] = size(testDataProj);
    N_C = length(unique(testLabel));%# of class
    N_D_C = N_Train/N_C; %# of samples per class
    D = zeros(N_Train,1); %distance matrix
    res = zeros(N_Test,1);
    
    %Calculate the distance between the testing data and every training
    %data
    for i = 1:N_Test
        for j = 1:N_Train
            tmp = (testDataProj(:,i) - trainDataProj(:,j))'*(testDataProj(:,i) - trainDataProj(:,j));
            D(j) = sqrt(tmp);
        end
        [~,index] = sort(D); %sort in increasing order
        L = floor((index-1)./N_D_C)+1; %convert to the label
        vote = zeros(N_Train,1);
        for k = 1:K %take the vote on the first K distance
            vote(L(k)) = vote(L(k)) + 1; 
        end
        max_v = max(vote); %take the maximun vote
        r = find(vote == max_v); %if there are multiple maximum values
        for h = 1:N_Train 
            if ismember(L(h),r) %take the one contains the minimum distance
                break 
            end
        end
        res(i) = L(h);
    end
    
    %Calculate the accuracy
    correct = 0;
    for i = 1:N_Test
        if res(i) == testLabel(i)
            correct = correct + 1;
        end
    end
    accuracy = correct / N_Test;
end