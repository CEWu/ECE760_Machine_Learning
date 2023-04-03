function [Z, params, Y] = deMeanPrinCompAnalysis(X, d)

  % First normalize the data
  meanX = mean(X);
  X_ = bsxfun(@minus, X, mean(X));

  % Now apply PCA
  [Z, params, Y_] = buggyPrinCompAnalysis(X_, d);

  % Now reconstruct
  Y = bsxfun(@plus, Y_, meanX);
  params.meanX = meanX; 

end

