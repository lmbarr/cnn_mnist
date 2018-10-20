function [conv1] = dimensionesActivaciones(Kernels, Pool)
conv1DimH = (imageDimH - Kernels.k1(1) + 2*(Kernels.k1(5)))/Kernels.k1(4) + 1;
conv1DimV = (imageDimV - Kernels.k1(1) + 2*(Kernels.k1(5)))/Kernels.k1(4) + 1;
conv1Prof = Kernels.k1(3);

pool1DimH = (conv1DimH - Pool.p1(1))/Pool.p1(2) + 1;
pool1DimV = (conv1DimV - Pool.p1(1))/Pool.p1(2) + 1;
pool1Prof = conv1Prof;

conv2DimH = (pool1DimH - Kernels.k2(1) + 2*(Kernels.k2(5)))/Kernels.k2(4) + 1;
conv2DimV = (pool1DimV - Kernels.k2(1) + 2*(Kernels.k2(5)))/Kernels.k2(4) + 1;
conv2Prof = Kernels.k2(3);

pool2DimH = (conv2DimH - Pool.p2(1))/Pool.p2(2) + 1;
pool2DimV = (conv2DimV - Pool.p2(1))/Pool.p2(2) + 1;
pool2Prof = conv2Prof;

conv3DimH = (pool2DimH - Kernels.k3(1) + 2*(Kernels.k3(5)))/Kernels.k3(4) + 1;
conv3DimV = (pool2DimV - Kernels.k3(1) + 2*(Kernels.k3(5)))/Kernels.k3(4) + 1;
conv3Prof = Kernels.k3(3);

conv4DimH = (conv3DimH - Kernels.k4(1) + 2*(Kernels.k4(5)))/Kernels.k4(4) + 1;
conv4DimV = (conv3DimV - Kernels.k4(1) + 2*(Kernels.k4(5)))/Kernels.k4(4) + 1;
conv4Prof = Kernels.k4(3);

conv5DimH = (conv4DimH - Kernels.k5(1) + 2*(Kernels.k5(5)))/Kernels.k5(4) + 1;
conv5DimV = (conv4DimV - Kernels.k5(1) + 2*(Kernels.k5(5)))/Kernels.k5(4) + 1;
conv5Prof = Kernels.k5(3);

pool5DimH = (conv5DimH - Pool.p5(1))/Pool.p5(2) + 1;
pool5DimV = (conv5DimV - Pool.p5(1))/Pool.p5(2) + 1;
pool5Prof = conv5Prof;

aux = pool5DimH*pool5DimV*pool5Prof;
B6 = zeros(1, 1, 1, numUnidadesCapaEsc); %sin bias
r  = sqrt(6) / sqrt(numUnidadesCapaEsc + aux + 1);
W6 = rand(1, 1, aux, numUnidadesCapaEsc) * 2 * r - r;

B7 = zeros(1, 1, 1, numUnidadesCapaEsc);%sin bias
r  = sqrt(6) / sqrt(numUnidadesCapaEsc + numUnidadesCapaEsc + 1);
W7 = rand(1, 1, numUnidadesCapaEsc, numUnidadesCapaEsc) * 2 * r - r;

B8 = zeros(1, 1, 1, numClasses);
r  = sqrt(6) / sqrt(numClasses + numUnidadesCapaEsc + 1);
W8 = rand(1, 1, numUnidadesCapaEsc, numClasses) * 2 * r - r;

end