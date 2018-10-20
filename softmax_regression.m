function [f, g] = softmax_regression(theta, X, y) %cambien el reshape y el dato que devuelve
  
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  
  m=size(X,2);
  n=size(X,1); %incluye la fila de unos
  % theta is a vector;  need to reshape to n x num_classes.
  %theta = reshape(theta, n, []);
  num_classes = size(theta,2);

  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
  cl = (1: num_classes)';
  cl = repmat(cl, 1, m); %% matriz q contiene las clases en cada columna
  %y=1x60000
  yb = repmat(y, num_classes, 1); %%matriz de 10X60000 que contiene el ejemplo yi repetido en sus columnas
  den = repmat(sum(exp(theta' * X)), num_classes, 1); %%es la sumatorias de las columnas de exp(theta' * X) repetidas hascia abajo el numclases

  f = -(1.0 / m) * sum(sum((cl == yb) .* log(exp(theta' * X) ./ den ))) ;
  
  hT = exp(theta' * X) ./ den; %la columna es la salida de todas las clases para cada punto

  for i = 1: num_classes
      cla = i * ones(m, 1);
      g(:, i) = -(1.0 / m) *(X * ((cla == y') - hT(i, :)') )' ;
  end

  %g = g(:) ; % make gradient a vector for minFunc
end
