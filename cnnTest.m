%% STEP 1: Iniciar datos
clear all;
[Kernels Pool numUnidadesCapaEsc] = datosArquitecturaTest();
imageDimH = 32;
imageDimV = 32;
numClasses = 2;  % Number of classes 

[images1  y] = cargarSetTrain();
size(images1)
for i=1:length(y)
images(:,:,1,i) = images1(:,:,i);
end
y(y==0) = 2; % Remap 0 to 2
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.
% To speed up gradient checking, we will use a reduced network and
% a debugging data set

db_images = images(:,:,:,1:2);
db_labels = y(1:2);
db_theta = cnnInitParams(imageDimH, imageDimV, Kernels, Pool, numUnidadesCapaEsc, ...
    numClasses);

[cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,...
                            Kernels,Pool,numUnidadesCapaEsc);


% Check gradients
numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
                            db_labels,numClasses,Kernels,...
                            Pool,numUnidadesCapaEsc), db_theta);

% Use this to visually compare the gradients side by side
disp([numGrad grad]);

diff = norm(numGrad-grad)/norm(numGrad+grad);
% Should be small. In our implementation, these values are usually 
% less than 1e-9.
disp(diff); 

assert(diff < 1e-9,...
    'Difference too large. Check your gradient computation again');
    
