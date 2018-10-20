function [images, y] = cargarSetTrain() 
V = 32;
H = 32;
numImages1 = 500;
numImgNeg = round(numImages1*40/100);
numImgPos = round(numImages1*60/100);
numImages = numImgNeg + numImgPos;
idxPos = RandSimRepeticao(0, 10200, numImgPos);%60% de imagenes positivas sin repeticion
idxNeg = RandSimRepeticao(0, 43300, numImgNeg);%40% de imagenes positivas
size(idxPos);
size(idxNeg);
%% Extraccion de imagenes positivas
imgPos = zeros(V, H, numImgPos);%tam imagenXnumImagenPos
for i = 1: numImgPos
    idx = idxPos(i);
    idx = num2str(idx, '%05d');
    ruta = strcat('C:\Users\user\Desktop\Tesis\clasificador\Classification\Train\pos\',idx,'.png');
    I = imread(ruta);%64x32
    igs = mat2gray(I);
%     imshow(igs)
%     pause(0.5)
    imgPos(:, :, i) = imresize(igs, [32 32]);
end
%% Extraccion de imagenes negativas
imgNeg = zeros(V, H, numImgNeg);%tam imagenXnumImagenNeg
for i = 1: numImgNeg
    idx = idxNeg(i);
    idx = num2str(idx, '%05d');
    ruta = strcat('C:\Users\user\Desktop\Tesis\clasificador\Classification\Train\neg\',idx,'.png');
    I = imread(ruta);%64x32
    igs = mat2gray(I);
    imgNeg(:, :, i) = imresize(igs, [32 32]);
end
%% Mescla de las imagenes en un solo super dato images
% Agrege el procesamiento de las imagenes
images = zeros(V, H, numImages);
y = zeros(numImages, 1);
for i = 1: numImages
    clase = randi(2) - 1;%sortea la clase
    if clase == 0 && numImgNeg ~= 0
        num = randi(numImgNeg);
        images(:, :, i) = proImage(imgNeg(:, :, num));
        imgNeg(:, :, num) = [];
        y(i) = 0;
        numImgNeg = numImgNeg - 1;
   
    elseif clase == 1 && numImgPos ~= 0
        num = randi(numImgPos);
        images(:, :, i) = proImage(imgPos(:, :, num));
        imgPos(:, :, num) = [];
        y(i) = 1;
        numImgPos = numImgPos - 1;
     
    end
end

%% Mostrar imagenes
mostrar = false;
if mostrar
    for i = 1: numImages
        figure(1);
        imagesc(images(:,:, i))

        %imshow(images(:,:, i))
        proI = proImage(images(:,:, i));
        y(i)
        pause(2.5)
    end
end
end


