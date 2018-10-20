%Softmax Function.
%bottom is a 2d matrix: N x 1.
%top is a 2d matrix: M x 1.
%For formula, see https://en.wikipedia.org/wiki/Softmax_function.
%Formula: top_i=(exp(bottom_i-bottom_max)/(sum_i(exp(bottom_i-bottom_max))).
%Para todas las imagenes
%Cada columna es una salida
function top = softmax( bottom )
    numClasses = size(bottom, 1);
    bottomExp = exp(bottom-repmat(max(bottom), numClasses, 1));
    top = bottomExp ./ repmat(sum(bottomExp), numClasses, 1);
end
