function theta = cnnInitParams(imageDimH, imageDimV, Kernels, Pool, numUnidadesCapaEsc, ...
    numClasses)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%ki = [tamañoFiltro numCanales numFiltros slide paddig]
%pi = [tamañoPool stride]
W1 = 1e-1*rand(Kernels.k1(1), Kernels.k1(1), Kernels.k1(2), Kernels.k1(3));
W2 = 1e-1*rand(Kernels.k2(1), Kernels.k2(1), Kernels.k2(2), Kernels.k2(3));
W3 = 1e-1*rand(Kernels.k3(1), Kernels.k3(1), Kernels.k3(2), Kernels.k3(3));

B1 = zeros(1, 1, 1, Kernels.k1(3));
B2 = zeros(1, 1, 1, Kernels.k2(3));
B3 = zeros(1, 1, 1, Kernels.k3(3));


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


pool3DimH = (conv3DimH - Pool.p3(1))/Pool.p3(2) + 1;
pool3DimV = (conv3DimV - Pool.p3(1))/Pool.p3(2) + 1;
pool3Prof = conv3Prof;

aux = pool3DimH*pool3DimV*pool3Prof;
B4 = zeros(1, 1, 1, numUnidadesCapaEsc); %sin bias
r  = sqrt(6) / sqrt(numUnidadesCapaEsc + aux + 1);
W4= rand(1, 1, aux, numUnidadesCapaEsc) * 2 * r - r;

B5 = zeros(1, 1, 1, numUnidadesCapaEsc);%sin bias
r  = sqrt(6) / sqrt(numUnidadesCapaEsc + numUnidadesCapaEsc + 1);
W5 = rand(1, 1, numUnidadesCapaEsc, numUnidadesCapaEsc) * 2 * r - r;

B6 = zeros(1, 1, 1, numClasses);
r  = sqrt(6) / sqrt(numClasses + numUnidadesCapaEsc + 1);
W6 = rand(1, 1, numUnidadesCapaEsc, numClasses) * 2 * r - r;
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
% pause
theta = [W1(:) ; B1(:) ;  W2(:); B2(:); W3(:); B3(:);...
W4(:);B4(:); W5(:); B5(:); W6(:);  B6(:)];
end

