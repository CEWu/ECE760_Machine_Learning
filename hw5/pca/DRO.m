function [Z, params, Y] = DRO(X, d)

  % prelims
  n = size(X, 1);

  b = mean(X)';
  [U,S,V] = svd( bsxfun(@minus, X, b') , 'econ');
  plot(diag(S)),
  Ud = U(:, 1:d);
  Sd = S(1:d, 1:d);
  Vd = V(:, 1:d);
  A = (1/sqrt(n) * Sd * Vd')';

  % Z and Y
  Z = sqrt(n) * Ud;
  Y = bsxfun(@plus, Z*A', b');

  params.A = A;
  params.b = b;

end

