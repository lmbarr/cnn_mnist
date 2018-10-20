%% Convolution Neural Network Exercise

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.
clc

clear all;
% Configuration
imageDimH = 32;
imageDimV = 32;
numClasses = 2;  % Number of classes 

% Cargar data set train
[images1  y] = cargarSetTrain();
% images = randi([0 5],32,32,1,500);
% y = randi([0 1],500,1);
images = zeros(32,32,1,length(y));
% size(images1)
for i=1:length(y)
images(:,:,1,i) = images1(:,:,i);
end
y(y == 0) = 2; % Remap 0 to 2
% Initialize Parameters
[Kernels Pool numUnidadesCapaEsc] = datosArquitectura();
theta = cnnInitParams(imageDimH, imageDimV, Kernels, Pool, numUnidadesCapaEsc, ...
    numClasses);



%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 3; %estaba 3
options.minibatch = 10;%estaba 256
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,Kernels,...
                      Pool, numUnidadesCapaEsc),theta,images,y,options);

%%======================================================================
%% STEP 4: Test
 %Test the performance of the trained model using the test set. 
 
[testImages1, testLabels] = cargarSetTest();
testImages = zeros(32,32,1,length(testLabels));
for i=1:length(y)
testImages(:,:,1,i) = testImages1(:,:,i);
end
testLabels(testLabels == 0) = 2; % Remap 0 to 2
[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                Kernels,Pool,numUnidadesCapaEsc,true);

acc = sum(preds == testLabels) / length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',100*acc);
