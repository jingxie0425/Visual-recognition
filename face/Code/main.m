%% data.mat subject classification
clear;
load("../Data/data.mat");
numClass = 200;
trainNum = 200*2;
testNum = 200*1;
pixelNum = 24*21;
trainData = zeros(pixelNum,trainNum);
testData = zeros(pixelNum,testNum);
trainLabels = zeros(trainNum,1);
testLabels = zeros(testNum,1);
idxTr = 1;
idxTe = 1;
for n = 1:numClass
    j=1;
    trainData(:,idxTr) = reshape(face(:,:,3*n-j),[pixelNum,1]); %choose different training data
    trainLabels(idxTr) = n; 
    idxTr = idxTr +1;
    j=2;
    trainData(:,idxTr) = reshape(face(:,:,3*n-j),[pixelNum,1]); %choose different training data
    trainLabels(idxTr) = n; 
    idxTr = idxTr +1;

    j=0;
    testData(:,idxTe) = reshape(face(:,:,3*n-j),[pixelNum,1]); %take the remianing as testing data
    testLabels(idxTe) = n; 
    idxTe = idxTe + 1;

end

[testDataProj,trainDataProj] = PCA(trainData,testData,0.8); %PCA
% [testDataProj,trainDataProj] = LDA(trainData,testData,numClass) %LDA
% accuracy = Bayes(trainDataProj,testDataProj,testLabels) %BAYES
accuracy = K_NN(trainDataProj,testDataProj,testLabels,1) %KNN

%% illumination.mat
clear;
load("../Data/illumination.mat");
numClass = 68;
train_ends = 13;
trainNum = 68*train_ends;
testNum = 68*2;
pixelNum = 1920;
trainData = zeros(pixelNum,trainNum);
testData = zeros(pixelNum,testNum);
trainLabels = zeros(trainNum,1);
testLabels = zeros(testNum,1);
idxTr = 1;
idxTe = 1;
for n = 1:numClass
   for j = 1:train_ends
      trainData(:,idxTr) = illum(:,j,n);%choose different training data
      trainLabels(idxTr) = n; %record the label
      idxTr = idxTr +1;
   end
   for k = (train_ends+1):21
      testData(:,idxTe) = illum(:,k,n);%take the remianing as testing data
      testLabels(idxTe) = n; %record the label
      idxTe = idxTe + 1;
   end
end

% [Test_D_P,Train_D_P] = PCA(Train_Data,Test_Data,0.9); %PCA
[testDataProj,trainDataProj] = LDA(trainData,testData,numClass); %LDA
accuracy = Bayes(trainDataProj,testDataProj,testLabels); % BAYES
% accuracy = K_NN(Train_D_P,Test_D_P,L_Test,4); %KNN
display(accuracy)

%% pose.mat
clear;
load("../Data/pose.mat");
numClass = 68;
trainNum = 68*12;
testNum = 68*1;
pixelNum = 48*40;
trainData = zeros(pixelNum,trainNum);
testData = zeros(pixelNum,testNum);
trainLabels = zeros(trainNum,1);
testLabels = zeros(testNum,1);
idxTr = 1;
idxTe = 1;
for n = 1:numClass
   for j = 1:12
      trainData(:,idxTr) = reshape(pose(:,:,j,n),[pixelNum,1]);%choose different training data
      trainLabels(idxTr) = n;%record the label
      idxTr = idxTr +1;
   end
   for k = 13:13
      testData(:,idxTe) = reshape(pose(:,:,k,n),[pixelNum,1]);%take the remianing as testing data
      testLabels(idxTe) = n;%record the label
      idxTe = idxTe + 1;
   end
end

[testDataProj,trainDataProj] = PCA(trainData,testData,0.90); %PCA
% [Test_D_P,Train_D_P] = LDA(Train_Data,Test_Data,N_C); %LDA
accuracy = Bayes(trainDataProj,testDataProj,testLabels); %BAYES
% accuracy = K_NN(Train_D_P,Test_D_P,L_Test,4); %KNN
display(accuracy)

%% neutral vs. facial exp (1) Bayes+knn
clear;
load("../Data/data.mat"); % load the dataset
numClass = 2; % number of classes

totalPairNum = 200; % a pair is a subject with neutral and expression. 
trainPairNum = 200*0.9;
testPairNum = totalPairNum - trainPairNum;
trainPairs = 1:1:totalPairNum;
% generate the testPairNum random indices
randIdcs = randperm(length(trainPairs),testPairNum);
% initialize R to be the four numbers of trainPairID
testPairs = trainPairs(randIdcs);
% remove those testPairNum numbers from trainPairID
trainPairs(randIdcs) = [];

trainNum = 2*trainPairNum;
testNum = 2*testPairNum;
pixelNum = 24*21;
trainData = zeros(pixelNum,trainNum);
testData = zeros(pixelNum,testNum);
trainLabels = zeros(1,trainNum);
testLabels = zeros(1,testNum);
idxTr = 1;
idxTe = 1;
% generate training dataset
for n = trainPairs
      trainData(:,idxTr) = reshape(face(:,:,3*n-2),[pixelNum,1]);%first trainPair Neural
      trainLabels(idxTr) = 1; %label of Neutal is 1 
      idxTr = idxTr +1;
end
for n = trainPairs
      trainData(:,idxTr) = reshape(face(:,:,3*n-1),[pixelNum,1]);%first trainPair Expression
      trainLabels(idxTr) = 2; %label of Expression is 2
      idxTr = idxTr +1;
end
% generate testing dataset
for n = testPairs
      testData(:,idxTe) = reshape(face(:,:,3*n-2),[pixelNum,1]); %remaining Neural
      testLabels(idxTe) = 1; %label of Neutal is 1 
      idxTe = idxTe +1;
end
for n = testPairs
      testData(:,idxTe) = reshape(face(:,:,3*n-1),[pixelNum,1]); %remaining Epression
      testLabels(idxTe) = 2; %label of Expression is 2
      idxTe = idxTe +1;
end

[testDataProj,trainDataProj,dim] = PCA(trainData,testData,0.8); %PCA
% [testDataProj,trainDataProj] = LDA(trainData,testData,numClass); %LDA
% accuracy = Bayes(trainDataProj,testDataProj,testLabels) %BAYES
accuracy = K_NN(trainDataProj,testDataProj,testLabels,17) %KNN
display(accuracy)

%% neutral vs. facial exp (2) svm
clear;
load("../Data/data.mat"); % load the dataset
numClass = 2; % number of classes

totalPairNum = 200; % a pair is a subject with neutral and expression. 
trainPairNum = 150;
testPairNum = totalPairNum - trainPairNum;
trainPairs = 1:1:totalPairNum;
% generate the testPairNum random indices
randIdcs = randperm(length(trainPairs),testPairNum);
% initialize R to be the four numbers of trainPairID
testPairs = trainPairs(randIdcs);
% remove those testPairNum numbers from trainPairID
trainPairs(randIdcs) = [];

trainNum = 2*trainPairNum;
testNum = 2*testPairNum;
pixelNum = 24*21;
trainData = zeros(pixelNum,trainNum);
testData = zeros(pixelNum,testNum);
trainLabels = zeros(1,trainNum);
testLabels = zeros(1,testNum);
idxTr = 1;
idxTe = 1;
% generate training dataset
for n = trainPairs
      trainData(:,idxTr) = reshape(face(:,:,3*n-2),[pixelNum,1]);%first trainPair Neural
      trainLabels(idxTr) = 1; %label of Neutal is 1 
      idxTr = idxTr +1;
end
for n = trainPairs
      trainData(:,idxTr) = reshape(face(:,:,3*n-1),[pixelNum,1]);%first trainPair Expression
      trainLabels(idxTr) = -1; %label of Expression is -1
      idxTr = idxTr +1;
end
% generate testing dataset
for n = testPairs
      testData(:,idxTe) = reshape(face(:,:,3*n-2),[pixelNum,1]); %remaining Neural
      testLabels(idxTe) = 1; %label of Neutal is 1 
      idxTe = idxTe +1;
end
for n = testPairs
      testData(:,idxTe) = reshape(face(:,:,3*n-1),[pixelNum,1]); %remaining Epression
      testLabels(idxTe) = -1; %label of Expression is -1
      idxTe = idxTe +1;
end

[testDataProj,trainDataProj,dim] = PCA(trainData,testData,0.8); %PCA
% [testDataProj,trainDataProj] = LDA(trainData,testData,numClass); %LDA
% [~,accuracy,~,~,~,~] = Kernel_SVM(trainDataProj,testDataProj,trainLabels,testLabels,0.5); %Kernel SVM
[accuracy,r,alpha_opt,res] = Boosted_SVM(trainData,testData,trainLabels,testLabels,0.8,30); %Boosted SVM
display(accuracy)