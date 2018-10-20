function d = deltaConv(dant, W, kernel, convo)
    
    numImages = size(dant, 4);
    numCanales = size(W, 3);
    numFilters = size(W, 4);
    V = size(dant, 1) - 1 + kernel(1);
    H = size(dant, 2) - 1 + kernel(1);
    
    d = zeros(V, H, numCanales, numImages);

    for i = 1: numImages
        for j = 1: numCanales
            for k = 1: numFilters
                d(:, :, j, i) = conv2(dant(:, :, k, i), W(:, :, j, k), 'full') + d(:, :, j, i);
            end
        end
    end
   
    if exist('convo','var')
        d = d .* convo .*(1 - convo);
    else
        return;
    end
      
end