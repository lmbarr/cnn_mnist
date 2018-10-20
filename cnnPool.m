function pooledFeatures = cnnPool(poolDim, slide, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDimH = size(convolvedFeatures, 2);
convolvedDimV = size(convolvedFeatures, 1);
pooledFeatures = zeros((convolvedDimV-poolDim)/slide + 1, ...
        (convolvedDimH - poolDim) / slide + 1, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 

for imageNum = 1:numImages
  for filterNum = 1:numFilters
      aux = conv2(convolvedFeatures(:, :, filterNum, imageNum), ones(poolDim), 'valid');
      pooledFeatures(:, :, filterNum, imageNum) = aux(1:slide:end, 1:slide:end) ./ (poolDim .^ 2);
  end
end

end

