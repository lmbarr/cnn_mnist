% T inclye el bias colocado en la pirmera fila
% dpost es el delta posterior
% a_act activaciones de capa actual de la forma
%numUnidades+1 X numImagenes
function d = delta(T, dpost, a_act)
    
    if ~exist('a_act','var')
        d = T * dpost;
        return;
    else
        a_act = [ones(1, size(a_act, 2)) ; a_act];
        d = T * dpost .* a_act .* (1 - a_act);
    end
end
