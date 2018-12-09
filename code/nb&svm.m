% EC503: Learning from Data/Project
% Author: Yanxing Zhang,Tong Ye,Zhengxiang Zhong
% This file is used to implement naive bayes and svm classification for the given dataset.
% Input:Training Dataset,Testing Dataset,naivebayes distrubitionname,
% multi-class svm method (ova,ovo)
% Output:CCR confusion matrixs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%For letter-recognition dataset
clc;clear;
load('letter-recognition.mat')
tic
[ccr,conf_matrix] = naivebayes(X_train,X_test,cell2mat(Y_train),cell2mat(Y_test),0,'normal')
toc
tic
ccr2 = svmmm(X_train,X_test,cell2mat(Y_train),cell2mat(Y_test),'onevsone','sd');
toc
tic
ccr1 = svmmm(X_train,X_test,cell2mat(Y_train),cell2mat(Y_test),'onevsall','sd');
toc 


%%
%For Mnist dataset
clc;clear;
load('data_mnist_train.mat')
load('data_mnist_test.mat')
tic
[ccr,conf_matrix] = naivebayes(X_train,X_test,Y_train,Y_test,0,'mn')
toc
tic
ccr2 = svmmm(X_train,X_test,Y_train,Y_test,'onevsone','sd');
toc

tic
ccr1 = svmmm(X_train,X_test,Y_train,Y_test,'onevsall','sd');
toc 

%%
%For adult dataset
clc;clear;
load census1994
X_train = adultdata(:,1:14);X_test = adulttest(:,1:14);
Y_train = adultdata(:,15);Y_test = adulttest(:,15);
tic
[ccr,conf_matrix,ypredict] = naivebayes(X_train,X_test,Y_train,Y_test,1,'normal');
toc
tic
ccr1 = svmmm(X_train,X_test,Y_train,Y_test,'onevsall','ad');
toc 
tic
ccr2 = svmmm(X_train,X_test,Y_train,Y_test,'onevsone','ad');
toc


%Helper function of Naive Bayes and SVM
function [ccr,conf_matrix,ypredict] = naivebayes(xtrain,xtest,ytrain,ytest,disnorm,method)

if (disnorm == 1)
    Nb = fitcnb(xtrain,ytrain);
else
    Nb = fitcnb(xtrain,ytrain,'DistributionNames',method);
end
ypredict = predict(Nb,xtest);
if (disnorm == 1)
    [conf_matrix,~] = confusionmat(ypredict,ytest.salary);
    ccr = sum(ypredict == ytest.salary)/size(ytest,1);
else
    [conf_matrix,~] = confusionmat(ypredict,ytest);
    ccr = sum(ypredict == ytest)/size(ytest,1);
end


end

function ccr = svmmm(xtrain,xtest,ytrain,ytest,method,s)
if (s == 'ad')
    Svm = fitcecoc(xtrain,ytrain,'Coding',method);
    ypredict = predict(Svm,xtest);
    ccr = sum(ypredict == ytest.salary)/size(ytest.salary,1);
else
    Svm = fitcecoc(xtrain,ytrain,'Coding',method);
    ypredict = predict(Svm,xtest);
    ccr = sum(ypredict == ytest)/size(ytest,1);
end
end


