function Dim = dimActivaciones(imageDimH, imageDimV, Kernels, Pool, numUnidadesCapaEsc)
%% Capa 1
conv1DimH = (imageDimH - Kernels.k1(1) + 2*(Kernels.k1(5)))/Kernels.k1(4) + 1;
conv1DimV = (imageDimV - Kernels.k1(1) + 2*(Kernels.k1(5)))/Kernels.k1(4) + 1;
conv1Prof = Kernels.k1(3);

field1 = 'conv1Dim';  value1 = [conv1DimV conv1DimH conv1Prof];
%%
pool1DimH = (conv1DimH - Pool.p1(1))/Pool.p1(2) + 1;
pool1DimV = (conv1DimV - Pool.p1(1))/Pool.p1(2) + 1;
pool1Prof = conv1Prof;
field2 = 'pool1Dim';  value2 = [pool1DimV pool1DimH pool1Prof];
%% Capa 2
conv2DimH = (pool1DimH - Kernels.k2(1) + 2*(Kernels.k2(5)))/Kernels.k2(4) + 1;
conv2DimV = (pool1DimV - Kernels.k2(1) + 2*(Kernels.k2(5)))/Kernels.k2(4) + 1;
conv2Prof = Kernels.k2(3);

field3 = 'conv2Dim';  value3 = [conv2DimV conv2DimH conv2Prof];

% pool2DimH = (conv2DimH - Pool.p2(1))/Pool.p2(2) + 1;
% pool2DimV = (conv2DimV - Pool.p2(1))/Pool.p2(2) + 1;
% pool2Prof = conv2Prof;
%% Capa 3
conv3DimH = (conv2DimH - Kernels.k3(1) + 2*(Kernels.k3(5)))/Kernels.k3(4) + 1;
conv3DimV = (conv2DimV - Kernels.k3(1) + 2*(Kernels.k3(5)))/Kernels.k3(4) + 1;
conv3Prof = Kernels.k3(3);

field4 = 'conv3Dim';  value4 = [conv3DimV conv3DimH conv3Prof];

%% 
pool3DimH = (conv3DimH - Pool.p3(1))/Pool.p3(2) + 1;
pool3DimV = (conv3DimV - Pool.p3(1))/Pool.p3(2) + 1;
pool3Prof = conv3Prof;

field5 = 'pool3Dim';  value5 = [pool3DimV pool3DimH pool3Prof];
%% 
% aux = pool3DimH*pool3DimV*pool3Prof;
% B4 = zeros(1, 1, 1, numUnidadesCapaEsc); %sin bias
% r  = sqrt(6) / sqrt(numUnidadesCapaEsc + aux + 1);
% W4= rand(1, 1, aux, numUnidadesCapaEsc) * 2 * r - r;
% 
% B5 = zeros(1, 1, 1, numUnidadesCapaEsc);%sin bias
% r  = sqrt(6) / sqrt(numUnidadesCapaEsc + numUnidadesCapaEsc + 1);
% W5 = rand(1, 1, numUnidadesCapaEsc, numUnidadesCapaEsc) * 2 * r - r;
% 
% B6 = zeros(1, 1, 1, numClasses);
% r  = sqrt(6) / sqrt(numClasses + numUnidadesCapaEsc + 1);
% W6 = rand(1, 1, numUnidadesCapaEsc, numClasses) * 2 * r - r;

Dim = struct(field1,value1,field2,value2, field3,value3, field4,value4, field5,value5);

end