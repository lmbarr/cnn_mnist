%ki = [tamañoFiltro numCanales numFiltros slide paddig]
%pi = [tamañoPool stride]
function [Kernels Pool numUnidadesEsc] = datosArquitectura

    
    numUnidadesEsc = 4096;
    field1 = 'k1';  value1 = [5 1 100  1 0];
    field2 = 'k2';  value2 = [3 100 384 1 0];
    field3 = 'k3';  value3 = [3 384 256 1 0];
    
    Kernels = struct(field1,value1,field2,value2,field3,value3);
    field1 = 'p1';  value1 = [2 2];
    field2 = 'p3';  value2 = [5 1];

    Pool = struct(field1,value1,field2,value2);
end