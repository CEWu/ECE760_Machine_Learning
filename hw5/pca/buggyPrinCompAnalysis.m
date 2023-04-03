function [Z, params, Y] = buggyPrinCompAnalysis(X, d)
% Z are the d dimensional representations. 
% Y are the reconstructions

  D = size(X, 2);
  [U, S, V] = svd(X, 'econ');
  Vd = V(:,1:d);

  params.Vd = Vd;
  Z = X * Vd;
  Y = Z * Vd';

end

