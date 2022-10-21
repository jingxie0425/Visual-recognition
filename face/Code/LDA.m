function [testDataProj,trainDataProj] = LDA(trainData,testData,classNum)
%
% This function implement the Linear Discriminant Analysis.
%
% [testDataProj,trainDataProj] = LDA(trainData,testData,N_C)
%
% trainData - The training data (Must be reshaped to a vector)
% testData - The testing data (Must be reshaped to a vector)
% classNum - Number of classes in the dataset.
% testDataProj - The projected training data (each column is an observation)
% trainDataProj - The projected testing data (each column is an observation)
%
% Jing Xie
% 16/11/2020
%
    %Get the size of the data and initialize the parameters
    [pixelNum,trainNum] = size(trainData);
    [~,testNum] = size(testData);
    samNumPC = trainNum/classNum; %# of samples per class
    
    %Calculate the mean value
    mu = zeros(pixelNum,classNum); % mean value of each class
    mu_all = zeros(pixelNum,1); %overall mean value
    for j = 1:classNum % mean value of each class
        for i = 1+samNumPC*(j-1):samNumPC*j
            mu(:,j) = mu(:,j) + trainData(:,i); 
        end
        mu(:,j) = mu(:,j)/samNumPC;
    end
    for i = 1:trainNum %overall mean value
        mu_all = mu_all + trainData(:,i);
    end
    mu_all = mu_all/trainNum;
    
    %Calculate between-scatter matrix
    sigmaBt = zeros(pixelNum,pixelNum);
    for j = 1:classNum
        tmp = mu(:,j) - mu_all;
        sigmaBt = sigmaBt + (tmp * tmp.');
    end
     
    %Calculate within-scatter matrix
    sigmaM = zeros(pixelNum,pixelNum);
    sigmaWth = zeros(pixelNum,pixelNum);
    for j = 1:classNum
        start = 1+samNumPC*(j-1);
        for k = 1:samNumPC
            diff = trainData(:,start+k-1) - mu(:,j);
            sigmaM = sigmaM + diff*diff.';
        end
        sigmaM = sigmaM + 1 * eye(pixelNum);%matrix singularity
        sigmaWth = sigmaWth + sigmaM;
    end
    
    %Take the first # classes - 1 eigenvectors
    [eigenVec,~] = eigs(sigmaBt,sigmaWth,classNum-1);

    %Project the training and testing data respectively
    trainDataProj = zeros(classNum-1,trainNum);
    testDataProj = zeros(classNum-1,testNum);
    for i = 1:trainNum
        trainDataProj(:,i) = eigenVec.' * trainData(:,i);
    end
    for i = 1:testNum
        testDataProj(:,i) = eigenVec.' * testData(:,i); 
    end
end