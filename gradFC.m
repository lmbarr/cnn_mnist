% Gradiente de fully conected layer
% en las columnas estan las salidas, una 
% columa por imagen
function [Wgrad, bgrad] = gradFC(a, d)
    W = zeros(size(d,1), size(a, 1));
    m = size(a,2);
    for i = 1: m
        W = d(:,i) * a(:,i)' + W;
    end
    Wgrad = zeros(1, 1, size(a,1), size(d,1));
    bgrad = zeros(1,1,1,size(d,1));
    Wgrad(1,1,:,:) = (1.0 / m) * W';
    bgrad(1,1,1,:) = (1.0 / m) * sum(d, 2); %es vector columna
end
