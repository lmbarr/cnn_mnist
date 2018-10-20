function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                Kernels,Pool,numUnidadesCapaEsc,pred)
% Calculate cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numCanales x
%           numImagenes  array
%  numClasses -  number of classes to predict
%  Kernels     Contienes informacion de los fltros de todas la capas de
%           convoluvion
%  Pool contiene informacion de los pooling realizado en las capas
%           especificado en la arquitectura
%  pred       -  boolean only forward propagate and return
%                predictions

%  Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

if ~exist('pred','var')
    pred = false;
end;

imageDimH = size(images, 2); % width of image
imageDimV = size(images, 1);
numImages = size(images, 4); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias

[W1, W2, W3, W4, W5, W6...
    B1, B2, B3, B4, B5, B6] = cnnParamsToStack(theta,imageDimH,imageDimV,Kernels,...
                                 Pool, numUnidadesCapaEsc, numClasses); %estilo alexnet

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
W1_grad = zeros(size(W1));B1_grad = zeros(size(B1));
W2_grad = zeros(size(W2));B2_grad = zeros(size(B2));
W3_grad = zeros(size(W3));B3_grad = zeros(size(B3));
W4_grad = zeros(size(W4));B4_grad = zeros(size(B4));
W5_grad = zeros(size(W5));B5_grad = zeros(size(B5));
W6_grad = zeros(size(W6));B6_grad = zeros(size(B6));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

conv1 = cnnConvolve(Kernels.k1(1), Kernels.k1(3), images, W1, B1);
%devuelve convDimV X convDimH X numFilters X numImages;
pool1 = cnnPool(Pool.p1(1), Pool.p1(2), conv1);
conv21 = cnnConvolve(Kernels.k2(1), Kernels.k2(3), pool1, W2, B2);
conv3 = cnnConvolve(Kernels.k3(1), Kernels.k3(3), conv21, W3, B3);
pool3 = cnnPool(Pool.p3(1), Pool.p3(2), conv3);
%Comienza la red neuronal convencional
%Cada columan en pool3 es una imagen no tiene bias
pool3 = reshape(pool3, size(pool3,1)*size(pool3,2)*size(pool3,3), numImages);
z4 = fc(pool3, W4, B4); 
a4 = sigmoid(z4);
Dima4 = size(a4);
z5 = fc(a4, W5, B5);
a5 = sigmoid(z5);
Dima5 = size(a5);
z6 = fc(a5, W6, B6);
probs = softmax(z6);

%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective(cost).  Store your
%  results in cost.
%Se agrega el bias a los pesos y la fila de unos X0 a las entrada(datos)
%theta entra a la funcion de la forma numfeaturesXnumclases...inlcuye bias
%La entrada a la funcion softmax tiene que ser el resultado del sigmoid(o relu)
[~,~,N,M] = size(W6);
weightFlattened = reshape(W6, [N, M]);
Bsoftmax = reshape(B6, [M, 1]);
Wsoftmax = weightFlattened';
%a5 la entrada se ubica en las columnas
%Sin la transposicion la primera columna es el bias
%Wsoftmax=numclassesXnumHiddenUnits
[cost, grad] = softmax_regression([Bsoftmax Wsoftmax]', [ones(1, numImages); a5], labels') ;
B6_grad(1, 1, :, :) = grad(1, :);
W6_grad(1, 1, :, :)= grad(2:end, :);

%% Predicciones 
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.
%  Use the kron function and a matrix of ones to do this upsampling
%  quickly.
%% delta softmax 7
yb = full(sparse(labels, 1:length(labels), 1));  %numClasses*numImages
d7 = probs - yb; %%numClasses*numImages

%% delta de capa escondida 6
T = [Bsoftmax Wsoftmax]'; %4097X2 %numHiddenUnitsXnumClases
d6 = delta(T, d7, a5);

%% delta de capa escondida 5
[~,~,N,M] = size(W5);
weightFlattened = reshape(W5, [N, M]);
B = reshape(B5, [M, 1]);
W = weightFlattened';
T = [B W]'; %4097X4096 %numHiddenUnitsXnumClases
d5 = delta(T, d6(2:end,:), a4);

%% delta capa 4
[~,~,N,M] = size(W4);
weightFlattened = reshape(W4, [N, M]);
B = reshape(B4, [M, 1]);
W = weightFlattened';
T = [B W]'; %4097X4096 %numHiddenUnitsXnumClases
d4n = delta(T, d5(2:end,:));%cambie d6 a d5

Dim = dimActivaciones(imageDimH, imageDimV, Kernels, Pool, numUnidadesCapaEsc);
%9216xnumImages------> 6X6X256XnumImages
d4FormaImagen = reshape(d4n(2:end, :), Dim.pool3Dim(1), Dim.pool3Dim(2), Kernels.k3(3), numImages);
delta_unpool = unPooling(d4FormaImagen, Pool.p3);%%10x10x256xnumImages upasample(W * d)

d4 = delta_unpool .* conv3 .* (1 - conv3); %%10x10x256xnumImages upsample(W * d)* f'()

%% delta capa 3
d3 = deltaConv(d4, W3, Kernels.k3, conv21);%%duda mando las activaciones del pool y antes de pool


%% delta de capa 2
d2 = deltaConv(d3, W2, Kernels.k2);%%no se multiplica por la activacion 

%%porque hay que hacer primero unsampling
d2_unpool = unPooling(d2, Pool.p1);
d2 = d2_unpool .* conv1 .* (1 - conv1);
%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.
%% Gradiente delta2

[W1_grad, B1_grad] = gradConv(images, d2, Kernels.k1);

%% Gradiente delta3
[W2_grad, B2_grad] = gradConv(pool1, d3, Kernels.k2);

%% Gradiente delta4
[W3_grad, B3_grad] = gradConv(conv21, d4, Kernels.k3);

%% Gradiente delta4
[W4_grad, B4_grad] = gradFC(pool3, d5(2:end,:));

%% Gradiente delta5
[W5_grad, B5_grad] = gradFC(a4, d6(2:end,:));

%% Unroll gradient into grad vector for minFunc

grad = [W1_grad(:) ; B1_grad(:) ;  W2_grad(:); B2_grad(:); W3_grad(:); B3_grad(:);...
W4_grad(:);B4_grad(:); W5_grad(:); B5_grad(:); W6_grad(:);  B6_grad(:)];
end
