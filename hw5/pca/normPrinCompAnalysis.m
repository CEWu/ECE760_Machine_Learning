function [Z, params, Y] = normPrinCompAnalysis(X, d)

  % First normalize the data
  meanX = mean(X);
  stdX = std(X);
  X_ = bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), stdX);

  % Now apply PCA
  [Z, params, Y_] = buggyPrinCompAnalysis(X_, d);

  % Now reconstruct
  Y = bsxfun(@plus, bsxfun(@times, Y_, stdX), meanX);
  params.meanX = meanX; 
  params.stdX = stdX;

end

