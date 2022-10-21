function [testDataProj,trainDataProj,n] = PCA(trainData,testData,lamda)
%
% This function implement the Principle Component Analysis.
%
% [testDataProj,trainDataProj] = PCA(trainData,testData,lamda)
%
% trainData is The training data (Must be reshaped to a vector)
% testData is The testing data (Must be reshaped to a vector)
% Lamda is A designed parameter to decide the number of dimensions.
%         Lager lamda meas more dimensions (The values should be within (0,1))
% testDataProj is The projected training data (each column is an observation)
% trainDataProj is The projected testing data (each column is an observation)
%
% Jing Xie
% 16/11/2020
%
    %Get the number of training data and the number of pixels
    [pixelNum,trainNum] = size(trainData);
    %Get the number of test data
    [~,testNum] = size(testData);
    
    %Center the training data
    dataCen = zeros(pixelNum,trainNum);
    for i = 1:pixelNum
        dataCen(i,:) = trainData(i,:) - mean(trainData(i,:));
    end
    
    %Calculate the sigma matrix
    sigma = zeros(pixelNum,pixelNum);
    for j = 1:trainNum
        sigma = sigma + dataCen(:,j)*(dataCen(:,j))';
    end
    sigma = sigma / (trainNum);
    
    %Calculate the eigenvalues and eigenvectors
    [eigenVec,eigenVal] = eig(sigma);
    eValReVec = eigenVal*ones(pixelNum,1); %reshape it into a vector
    [eValReVecSt,index] = sort(eValReVec,'descend'); %sort in decreasing order
    
    %Decide how many dimensions should take
    eigenSum = sum(eValReVecSt);%total sum of all the eigenvalues
    currentSum = 0;
    for n = 1:pixelNum
        currentSum = currentSum + eValReVecSt(n);
        if currentSum >= lamda * eigenSum %when the sum reached the threshold
            break %stop the iteration
        end
    end
    u = zeros(pixelNum,n);
    for m = 1:n
        u(:,m) = eigenVec(:,index(m));
    end
    
    %Project the training and testing data respectively
    trainDataProj = zeros(n,trainNum);
    testDataProj = zeros(n,testNum);
    for j = 1:trainNum
        trainDataProj(:,j) = u.' * trainData(:,j);
    end
    for i = 1:testNum
        testDataProj(:,i) = u.' * testData(:,i); 
    end
end
