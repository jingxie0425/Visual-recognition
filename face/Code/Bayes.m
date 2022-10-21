function accuracy = Bayes(trainDataProj,testDataProj,testLabel)
%
% This function implement the Bayes' classifier using the Maximum Likelihood.
%
% accuracy = Bayes(Train_D_P,Test_D_P,L_Test)
%
% trainDataProj - training data (each column is an observation)
% testDataProj - testing data (each column is an observation)
% testLabel - true label of testing data
% accuracy - accuracy of the classifier (the value is within (0,1))
%
% Jing Xie
% 15/11/2020
%
    %Get the size of the data and initialize the parameters
    [pixelNum,numData] = size(trainDataProj);
    [~,testNum] = size(testDataProj);
    numClass = length(unique(testLabel)); % number of classes
    M = zeros(pixelNum,numClass); % mean value matrix
    c = zeros(pixelNum,pixelNum,numClass); %covariance matrix
    CInv = zeros(pixelNum,pixelNum,numClass); %inverse covariance matrix
    numDataPC = numData/numClass; % # of data per class
    
    %Calculate the mean and covariance of the training data
    for i = 1:numClass
        start = 1+numDataPC*(i-1);
        stop = numDataPC*i;
        M(:,i) = mean(trainDataProj(:,start:stop),2);
        c(:,:,i) = cov(transpose(trainDataProj(:,1+numDataPC*(i-1):numDataPC*i)));
        c(:,:,i) = c(:,:,i) + 1 * eye(pixelNum); % covariance matrix singularity
        CInv(:,:,i) = inv(c(:,:,i));
    end
    
    %Calculate the probabilities of each class for every testing data
    P_0 = zeros(numClass,1);
    P_1 = zeros(pixelNum,numClass);
    for j = 1:numClass %Terms that only depend on the class
        P_0(j) = -0.5 * M(:,j)'*CInv(:,:,j)*M(:,j)-0.5*log(det(c(:,:,j)));
        P_1(:,j) = CInv(:,:,j) * M(:,j);
    end
    res = zeros(testNum,1);
    for i = 1:testNum
        max_prob = -inf(1);
        for j = 1:numClass
            tmp =  -0.5 * testDataProj(:,i)' * CInv(:,:,j) * testDataProj(:,i);
            prob = tmp + P_0(j) + P_1(:,j)'*testDataProj(:,i);
            if prob > max_prob %take the maximum probability
                max_prob = prob;
                res(i) = j;
            end
        end
    end
    
    %Calculate the accuracy
    correct = 0;
    for i = 1:testNum
        if res(i) == testLabel(i)
            correct = correct + 1;
        end
    end
    accuracy = correct / testNum;
end