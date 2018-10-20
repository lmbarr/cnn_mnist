function delta_pool = unPooling(delta, Pool)
    numFilters = size(delta, 3);
    numImages = size(delta, 4);
    H = (size(delta, 2) - 1)*Pool(2)+Pool(1);
    V = (size(delta, 1) - 1)*Pool(2)+Pool(1);
    
    delta_pool = zeros(H, V, numFilters, numImages);
    if Pool(2) == 1
        %% unPooling para slide 1 y de cualquier tamaño de ventana
        for i = 1: numImages
            for j = 1 : numFilters
                delta_pool(:, :, j, i) = (1.0 / Pool(1) ^ 2) * conv2(delta(:, :, j, i), ones(Pool(1)),'full');
            end
        end
    else
        for i = 1: numImages
            for j = 1: numFilters
                delta_pool(:, :,j, i) = (1.0 / Pool(1) ^ 2) * kron(delta(:, :, j, i), ones(Pool(1)));
            end
        end
    end

end