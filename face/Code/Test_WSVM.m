function result = Test_WSVM(x,y,alpha,bias,testData,testLabel)
%
% Weak SVM is tested in this function
%
% result = Test_WSVM(x,y,alpha,bias,testData,testLabel)
%
% x is support vectors
% y is the corresponding true label of support vectors
% alpha is the weights of different support vectors in the final classifier
% bias is bias term in final classifier
% Test_Data is the data to test the weak SVM
% testLabel is the correponding true label of the data
% result is the resulting value before take the sign function
%
% Jing
% 16/11/2020
%
    result = zeros(1,length(testLabel));
    K_Test = kernel(x,testData,'l',100);
    for h = 1:length(testLabel)
        result(h) = sum(alpha.*y.*K_Test(h,:))+bias;
    end
end