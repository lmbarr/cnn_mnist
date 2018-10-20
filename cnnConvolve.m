function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)
% Convolucion con el W que entra rotado 180(rot90(a,2)), por defecto
% padding  0 y slide 1 ademas incluye la activacion
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(images, 4);
numCanales = size(images, 3);
imageDimV = size(images, 1);
imageDimH = size(images, 2);
convDimH = imageDimH - filterDim + 1;
convDimV = imageDimV - filterDim + 1;

convolvedFeatures = zeros(convDimV, convDimH, numFilters, numImages);

% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 30 seconds 
%   Convolving with 5000 images should take around 2 minutes
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)


for imageNum = 1:numImages
  for filterNum = 1:numFilters
    convolvedImage = zeros(convDimV, convDimH);
    for channelNum = 1:numCanales
         % Obtain the feature (filterDim x filterDim) needed during the convolution
        filter = W(:, :, channelNum, filterNum);

        % Flip the feature matrix because of the definition of convolution, as explained later
        %filter = rot90(squeeze(filter),2);

        % Obtain the image
        im = images(:, :, channelNum, imageNum);

        % Convolve "filter" with "im", adding the result to convolvedImage
        % be sure to do a 'valid' convolution

        convolvedImage = conv2(im, rot90(filter, 2), 'valid') + convolvedImage;
        
        
    end
         convolvedImage = sigmoid(convolvedImage + b(filterNum) * ones(size(convolvedImage)));
         convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
   end

end


end

