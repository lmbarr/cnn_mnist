function [images, y] = cargarSetTest()
V = 32;
H = 32;
numImages1 = 1000;
numImgNeg = round(numImages1 * 40/100);
numImgPos = round(numImages1 * 60/100);
numImages = numImgPos + numImgNeg;
idxPos = RandSimRepeticao(0, 5943, numImgPos);%60% de imagenes positivas sin repeticion
idxNeg = RandSimRepeticao(0, 22049, numImgNeg);%40% de imagenes positivas
size(idxPos);
size(idxNeg);
%% Extraccion de imagenes positivas
imgPos = zeros(V, H, numImgPos);
for i = 1: numImgPos
    idx = idxPos(i);
    idx = num2str(idx, '%05d');
    ruta = strcat('C:\Users\user\Desktop\Tesis\clasificador\Classification\Test\pos\',idx,'.png');
    I = imread(ruta);%64x32
    igs = mat2gray(I);
%     imshow(igs)
%     pause(0.5)
    imgPos(:, :, i) = imresize(igs, [32 32]);
end
%% Extraccion de imagenes negativas
imgNeg = zeros(V, H, numImgNeg);
for i = 1: numImgNeg
    idx = idxNeg(i);
    idx = num2str(idx, '%05d');
    ruta = strcat('C:\Users\user\Desktop\Tesis\clasificador\Classification\Test\neg\',idx,'.png');
    I = imread(ruta);%64x32
    igs = mat2gray(I);
    imgNeg(:, :, i) = imresize(igs, [32 32]);
end
%% Mescla de las imagenes en un solo super dato images
%%Agrege procesamiento de las imagenes
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
   
    else
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
        imshow(images(:,:, i))
        imagesc(images(:,:, i))
        y(i)
        pause(3.5)
    end
end
end