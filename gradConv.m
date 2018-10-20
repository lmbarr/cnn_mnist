function [Wgrad, bgrad] = gradConv(a, d, kernel)
    numImages = size(a, 4);
    numFilters = kernel(3);
    numCanales = kernel(2);
    Wgrad = zeros(kernel(1), kernel(1), numCanales, numFilters);
    bgrad = zeros(1, 1, 1, numFilters);
    aux = zeros(kernel(1), kernel(1), numCanales);

    for k = 1: numFilters
        for i = 1: numImages
            for j = 1: numCanales
                aux(:, :, j) = conv2(a(:,:,j,i), rot90(d(:, :, k, i), 2), 'valid'); 
                %segun el curso ufldl 
            end
            Wgrad(:, :, :, k) = Wgrad(:, :, :, k) + aux;
            bgrad(1,1,1,k) = sum(sum(d(:, :, k, i))) + bgrad(1,1,1,k);
        end
    end
    Wgrad = (1.0 / numImages) .* Wgrad ;
    bgrad = (1.0 / numImages) .* bgrad ;
end