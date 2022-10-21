function [accuracy,r,alpha_opt,res] = Boosted_SVM(trainDataProj,testDataProj,trainLabel,testLabel,mu,T)
%
% Boosted SVM function
%
% [accuracy,r,alpha_opt,res] = Boosted_SVM(trainDataProj,testDataProj,trainLabel,testLabel,mu,T)
%
% trainDataProj is projected training data
% testDataProj is testing data projected
% trainLabel is true label of training data
% testLabel is true label of testing data
% mu is the parameter used in function WSVM
% T is number of weak SVM classifers
% accuracy is accurace of the classifier (a number between 0 and 1)
% r is perdicted label
% alpha_opt is optimal of wieghts of different weak SVM classifier
% res is the value before take the sign function
%
% Jing
% 16/11/2020
%
    %Get the size of data and initialization
    [~,N_Train] = size(trainDataProj);
    [~,N_Test] = size(testDataProj);
    Wn = ones(1,N_Train)/N_Train;
    res = zeros(T,length(testLabel));
    alpha_opt = zeros(1,T);
    alpha_opt(1) = 1;
    Wn_N = zeros(T,N_Train);
    
    %First weak SVM
    [H,x,y,alpha,bias] = WSVM(trainDataProj,trainLabel,Wn,mu);
    res(1,:) = Test_WSVM(x,y,alpha,bias,testDataProj,testLabel);
    h = H;
    Wn_N(1,:) = Wn;
    
    %Adaboost algorithm
    for t = 2:T
        Weighted_E = 0;
        for i = 1:length(h)
           Weighted_E = Weighted_E + Wn_N(t-1,i)*abs(sign(h(i))+trainLabel(i))/2; % calculate the weighted erroe
        end 
        for i = 1:length(h)
           Wn_N(t,i) =  Wn_N(t-1,i)*((1-Weighted_E)/Weighted_E)^(-0.5*trainLabel(i)*sign(h(i))); %update the weights of different data
        end
        z = sum(Wn_N(t,:));
        Wn_N(t,:) = Wn_N(t,:)./ z; % to make the weights as a distribution
        
        %Use the new wights to do weak SVM again 
        [h,x,y,alpha,bias] = WSVM(trainDataProj,trainLabel,Wn_N(t,:),mu);
        res(t,:) = Test_WSVM(x,y,alpha,bias,testDataProj,testLabel); % test the weak SVM
        
        %Find optimal alpha
        min_w_e = inf(1);
        for alpha_tmp = 0:0.05:1
            H_tmp = h.*alpha_tmp+H.*(1-alpha_tmp); %updata the result
            for i = 1:N_Train
                Weighted_E = Weighted_E + Wn_N(t,i)*abs(sign(H_tmp(i))+trainLabel(i))/2; %calculate the weighted error
            end
            if Weighted_E < min_w_e %take the alpha with minimum weighted error
                min_w_e = Weighted_E;
                alpha_opt(t) = alpha_tmp;
            end
            Weighted_E = 0; %reset the weighted error
        end
        
        H = alpha_opt(t)*h+(1-alpha_opt(t))*H; %update the result
    end
    r = zeros(size(res));
    for i = 1:t
       r(i,:) = alpha_opt(i)*res(i,:); %combine the result of different weak SVMs with different weights
       l_p = sign(sum(r));
    end
    
    %Calculate the accuracy
    correct = 0;
    for i = 1:N_Test
        if l_p(i) == testLabel(i)
            correct = correct + 1;
        end
    end
    accuracy = correct / N_Test;
end