function [res_val,accuracy,x,y,alpha,bias] = Kernel_SVM(trainDataProj,testDataProj,trainLabel,testLabel,C)
%
% This function implement kernel SVM with two different kernels(RBF and
% Polynomial).The change of kernel and parameters must be done in the
% function 
%
% [res_val,accuracy,x,y,alpha,bias] = Kernel_SVM(Train_D_P,Test_D_P,L_Train,L_Test,C)
%
% Train_D_P - training data
% Test_D_P - testing data
% L_Train - true label of training data
% L_Test - true label of testing data
% C - regualrization parameter control the misclassification of each
%     training sample. Larger C means a smaller-margin hyperplane
% res_val - the value before take the sign function
% accuracy - accuracy of the classifier
% x - support vectors
% y - the corresponding true label of support vectors
% alpha - the weights of different support vectors in the final classifier
% bias - bias term in final classifier
%
% Jing Xie
% 16/11/2020
%
    % Get the size of the data
    [~,n] = size(trainDataProj);
    [~,N_Test] = size(testDataProj);
    x = trainDataProj;
    y = trainLabel;
    
    %Initialize the parameters
    tol = 0.00001;
    alpha = zeros(1,n);
    bias = 0;
    it = 0;
    maxit = 100;

    %SMO algorithm
    while (it<maxit)
       it = it + 1;%record the iteration number
       %Initialize
       alphas_c = 0;
       N=length(y);
       K = kernel(x,x,'r',100); %calculate the kernel
       for i = 1:N
           Ei = sum(alpha.*y.*K(i,:)) + bias - y(i);
           y_kkt = Ei*y(i);
           if (alpha(i)<C && y_kkt<-tol) || (alpha(i)>0 && y_kkt>tol) %KKT conditions
               for j = [1:i-1,i+1:N] %optimize two alpha at a time
                  Ej = sum(alpha.*y.*K(j,:)) + bias - y(j);
                  ai_old = alpha(i); %record old ai
                  aj_old = alpha(j); %record old aj
                  if y(i)==y(j)
                     L = max(0,alpha(i)+alpha(j)-C);
                     H = min(C,alpha(j)+alpha(i));
                  else
                      L = max(0,alpha(j)-alpha(i));
                      H = min(C+alpha(j)-alpha(i),C);
                  end
                  if L == H %do nothing when L=H
                     continue 
                  end
                  eta = 2*K(i,j)-K(i:i)-K(j:j);
                  
                  %Updata alpha j
                  alpha(j) = alpha(j)+y(j)*(Ej-Ei)/eta;
                  if alpha(j)>H
                     alpha(j) = H;
                  elseif alpha(j) < L
                      alpha(j) = L;
                  end
                  if norm(alpha(j)-aj_old) < tol %stop this iteration when alpha j has very small change
                     continue 
                  end
                  
                  %Update alpha i
                  alpha(i) = alpha(i) - y(i) *y(j)*(alpha(j)-aj_old);
                  
                  %UpDATE bias term
                  bi = bias - Ei - y(i)*(alpha(i)-ai_old)*K(i,i)-y(j)*(alpha(j)-aj_old)*K(i,j);
                  bj = bias - Ej - y(i)*(alpha(i)-ai_old)*K(j,i)-y(j)*(alpha(j)-aj_old)*K(j,j);
                  if 0<alpha(i) && alpha(i)<C
                      bias = bi;
                  elseif 0<alpha(j) && alpha(j)<C
                      bias = bj;
                  else
                      bias = (bi+bj)/2;
                  end
                  alphas_c = alphas_c + 1; %plus one if alpha j and i been changed 
                end
           end
       end
       if alphas_c == 0 %break the for loop when no change
          break 
       end
       
       %Find the alphas with nonzero value and corresponding data and label
       index = find(alpha~=0);
       alpha = alpha(index);
       x = x(:,index);
       y = y(index);
    end
    
    %Calculate bias term of the final classifier
    nsv = length(y); %number of support vector
    bias = 0;
    Ksv = kernel(x,x,'r',100);
    for i = 1:nsv
        bias = bias + (y(i)-sum(y.*alpha.*Ksv(i,:)));
    end
    bias = bias/nsv;
    
    %Feed in the test data
    res = zeros(1,length(testLabel));
    res_val = zeros(1,length(testLabel));
    K_Test = kernel(x,testDataProj,'r',100);
    for h = 1:length(testLabel)
        res_val(h) = sum(alpha.*y.*K_Test(h,:))+bias;
        res(h) = sign(res_val(h)); %predicted label
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