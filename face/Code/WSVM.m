function [H,x,y,alpha,bias] = WSVM(trainData,trainLabel,weights,mu)
%
% Weak SVM function
%
% [H,x,y,alpha,bias] = WSVM(trainData,trainLabel,Wn,mu)
%
% trainData is all the training data
% trainLabel is correspongding true label
% weights is correspongding weights
% mu is a number between 0 and 1 decides the percentage of data
% H is the value before take the sign function
% x is support vectors
% y is the corresponding true label of support vectors
% alpha is the weights of different support vectors in the final classifier
% bias is bias term in final classifier
%
% Jing
% 16/11/2020
%
    %Get the size of data and initializations
    [N_Pixel,N_Train] = size(trainData);
    [Wn_sort,index] = sort(weights,'descend'); %sort the weight in decreasing order
    J = zeros(N_Train,1);
    C = 0.01;
    tmp = 0;
    
    %Choose part of the training data
    for i = 1:N_Train
       tmp = tmp + Wn_sort(i); 
       if tmp <= mu %take account the data when the sum is smaller than mu (the sum of total weights are 1)
          J(i) = index(i);
       else
           cnt = i-1;
           break
       end
       cnt = i;
    end
    trainDataNum = zeros(N_Pixel,cnt);
    trainLabelNum = zeros(1,cnt);
    Wn_N = zeros(1,cnt);
    for j = 1:cnt
        trainDataNum(:,j) = trainData(:,J(j)); %take the part of data
        trainLabelNum(:,j) = trainLabel(:,J(j)); %corresponding label
        Wn_N(j) = weights(J(j)); %corresponding weights
    end
    
    %Calculate the liner SVM
    [H,~,x,y,alpha,bias] = Kernel_SVM(trainDataNum,trainData,trainLabelNum,trainLabel,C);
end