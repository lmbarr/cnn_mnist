function [W1, W2, W3, W4, W5, W6, ...
    B1, B2, B3, B4, B5, B6] = cnnParamsToStack(theta,imageDimH,imageDimV,Kernels,...
                                 Pool, numUnidadesCapaEsc, numClasses)
    
% Converts unrolled parameters for a single layer convolutional neural
% network followed by a softmax layer into structured weight
% tensors/matrices and corresponding biases
%                            
% Parameters:
%  theta      -  unrolled parameter vectore
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  Wc      -  filterDim x filterDim x numFilters parameter matrix
%  Wd      -  numClasses x hiddenSize parameter matrix, hiddenSize is
%             calculated as numFilters*((imageDim-filterDim+1)/poolDim)^2 
%  bc      -  bias for convolution layer of size numFilters x 1
%  bd      -  bias for dense layer of size hiddenSize x 1

% outDimH = (imageDimH - filterDim + 1)/poolDim;
% outDimV = (imageDimV - filterDim + 1)/poolDim;
% hiddenSize = outDimH*outDimV*numFilters;
% 
% %% Reshape theta
% indS = 1;
% indE = filterDim^2*numFilters;
% Wc = reshape(theta(indS:indE),filterDim,filterDim,numFilters);
% indS = indE+1;
% indE = indE+hiddenSize*numClasses;
% Wd = reshape(theta(indS:indE),numClasses,hiddenSize);
% indS = indE+1;
% indE = indE+numFilters;
% bc = theta(indS:indE);
% bd = theta(indE+1:end);


indS = 1;
indE = Kernels.k1(1)* Kernels.k1(1)* Kernels.k1(2)* Kernels.k1(3);
W1 = reshape(theta(1:indE),Kernels.k1(1), Kernels.k1(1), Kernels.k1(2), Kernels.k1(3));

indS = indE + 1;
indE = indE + Kernels.k1(3);
B1 = reshape(theta(indS:indE),1, 1, 1, Kernels.k1(3));

indS = indE + 1;
indE = indE + Kernels.k2(1)* Kernels.k2(1)* Kernels.k2(2)* Kernels.k2(3);
W2 = reshape(theta(indS:indE),Kernels.k2(1), Kernels.k2(1), Kernels.k2(2), Kernels.k2(3));

indS = indE + 1;
indE = indE + Kernels.k2(3);
B2 = reshape(theta(indS:indE),1, 1, 1, Kernels.k2(3));



indS = indE + 1;
indE = indE + Kernels.k3(1)* Kernels.k3(1)* Kernels.k3(2)* Kernels.k3(3);
W3 = reshape(theta(indS:indE),Kernels.k3(1), Kernels.k3(1), Kernels.k3(2), Kernels.k3(3));

indS = indE + 1;
indE = indE + Kernels.k3(3);
B3 = reshape(theta(indS:indE),1, 1, 1, Kernels.k3(3));

% indS = indE + 1;
% indE = indE + Kernels.k4(1)*Kernels.k4(1)* Kernels.k4(2)* Kernels.k4(3);
% W4 = reshape(theta(indS:indE),Kernels.k4(1), Kernels.k4(1), Kernels.k4(2), Kernels.k4(3));

% indS = indE + 1;
% indE = indE + Kernels.k4(3);
% B4 = reshape(theta(indS:indE),1, 1, 1, Kernels.k4(3));
% 
% indS = indE + 1;
% indE = indE + Kernels.k5(1)* Kernels.k5(1)* Kernels.k5(2)* Kernels.k5(3);
% W5 = reshape(theta(indS:indE),Kernels.k5(1), Kernels.k5(1), Kernels.k5(2), Kernels.k5(3));
% 
% indS = indE + 1;
% indE = indE + Kernels.k5(3);
% B5 = reshape(theta(indS:indE),1, 1, 1, Kernels.k5(3));


conv1DimH = (imageDimH - Kernels.k1(1) + 2*(Kernels.k1(5)))/Kernels.k1(4) + 1;
conv1DimV = (imageDimV - Kernels.k1(1) + 2*(Kernels.k1(5)))/Kernels.k1(4) + 1;
conv1Prof = Kernels.k1(3);

pool1DimH = (conv1DimH - Pool.p1(1))/Pool.p1(2) + 1;
pool1DimV = (conv1DimV - Pool.p1(1))/Pool.p1(2) + 1;
pool1Prof = conv1Prof;

conv2DimH = (pool1DimH - Kernels.k2(1) + 2*(Kernels.k2(5)))/Kernels.k2(4) + 1;
conv2DimV = (pool1DimV - Kernels.k2(1) + 2*(Kernels.k2(5)))/Kernels.k2(4) + 1;
conv2Prof = Kernels.k2(3);

% pool2DimH = (conv2DimH - Pool.p2(1))/Pool.p2(2) + 1;
% pool2DimV = (conv2DimV - Pool.p2(1))/Pool.p2(2) + 1;
% pool2Prof = conv2Prof;

conv3DimH = (conv2DimH - Kernels.k3(1) + 2*(Kernels.k3(5)))/Kernels.k3(4) + 1;
conv3DimV = (conv2DimV - Kernels.k3(1) + 2*(Kernels.k3(5)))/Kernels.k3(4) + 1;
conv3Prof = Kernels.k3(3);

% conv4DimH = (conv3DimH - Kernels.k4(1) + 2*(Kernels.k4(5)))/Kernels.k4(4) + 1;
% conv4DimV = (conv3DimV - Kernels.k4(1) + 2*(Kernels.k4(5)))/Kernels.k4(4) + 1;
% conv4Prof = Kernels.k4(3);
% 
% conv5DimH = (conv4DimH - Kernels.k5(1) + 2*(Kernels.k5(5)))/Kernels.k5(4) + 1;
% conv5DimV = (conv4DimV - Kernels.k5(1) + 2*(Kernels.k5(5)))/Kernels.k5(4) + 1;
% conv5Prof = Kernels.k5(3);

pool3DimH = (conv3DimH - Pool.p3(1))/Pool.p3(2) + 1;
pool3DimV = (conv3DimV - Pool.p3(1))/Pool.p3(2) + 1;
pool3Prof = conv3Prof;

aux = pool3DimH*pool3DimV*pool3Prof;
dim6 = aux*numUnidadesCapaEsc;

indS = indE +1;
indE = indE + dim6;
W4 = reshape(theta(indS:indE),1, 1, aux, numUnidadesCapaEsc);

indS = indE + 1;
indE = indE + numUnidadesCapaEsc;
B4 = reshape(theta(indS:indE),1, 1, 1, numUnidadesCapaEsc); 

dim7 = numUnidadesCapaEsc* numUnidadesCapaEsc;
indS = indE + 1;
indE = indE + dim7;
W5 = reshape(theta(indS:indE),1, 1, numUnidadesCapaEsc, numUnidadesCapaEsc);

indS = indE + 1;
indE = indE + numUnidadesCapaEsc;
B5 = reshape(theta(indS:indE),1, 1, 1, numUnidadesCapaEsc);%sin bias

dim8 =  numUnidadesCapaEsc* numClasses;
indS = indE + 1;
indE = indE + dim8;
W6 = reshape(theta(indS:indE),1, 1, numUnidadesCapaEsc, numClasses);

indS = indE + 1;
indE = indE + numClasses;
B6 = reshape(theta(indS:indE),1, 1, 1, numClasses);
% size(W1)
% size(B1)
% 
% size(W2)
% size(B2)
% 
% size(W3)
% size(B3)
% size(W4)
% size(B4)
% 
% size(W5)
% size(B5)
% size(W6)
% size(B6)
% 
% pause;
end