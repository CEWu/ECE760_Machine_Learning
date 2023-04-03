function [Z, params, Y] = DRLV(X, d)

  % prelims
  NUM_EM_ITERS = 10;
  D = size(X, 2);
  n = size(X, 1);

  % Initialize using DRO
  [~, initParams, Y] = DRO(X, d);
  b = initParams.b; % this will also be the final b
  A = initParams.A;
  eta = sqrt( mean(mean( (Y-X).^2 )) );

  for emIter = 1:NUM_EM_ITERS
    [A, eta] = emFA(X, A, eta, b);
  end
  params.A = A;
  params.b = b;
  params.eta = eta;

  % Finally obtain Z and Y
  Z = bsxfun(@minus, X, b') * (( A*A' + eta^2 *eye(D)) \ A) ;
  Y = bsxfun(@plus, Z*A', b');

end


function [ANew, etaNew, RzxMeans] = emFA(X, AOld, etaOld, b)

  % prelims
  D = size(X, 2);
  d = size(AOld, 2);
  n = size(X, 1);

  % E-step
  K = AOld*AOld' + etaOld^2 * eye(D);
  Kinv = inv(K);
  RzxMeans = bsxfun(@minus, X, b') * Kinv * AOld;
  RzxVar = eye(d) - AOld' * Kinv * AOld;
  % Compute the following which will be useful too
  EAZ = RzxMeans * AOld';
  EAZ2 = sum( EAZ.^2 , 2) + sum(diag(AOld * RzxVar * AOld'));
  Xmb = bsxfun(@minus, X, b');

  % M-step
  % First A
  M1 = Xmb' * RzxMeans;
  M2 = RzxMeans' * RzxMeans + n * RzxVar;
  ANew = M1 / M2;
  % Now eta
  etaNew = sqrt( 1/(n*D) * ( norm(Xmb, 'fro')^2 ...
                 - 2 * sum(sum( (Xmb .* EAZ) ) ) ...
                 + sum(EAZ2) ) );

end

