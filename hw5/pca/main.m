% Starter code for problem 1.4
clear all;
close all

% Load the dataset
% T=readtable('data2D.csv');
% X=T{:,:};
% save('data2D.mat','X')
% load('data2D.mat');
T=readtable('data1000D.csv');
X=T{:,:};
save('data1000D.mat','X')
load('data1000D.mat');
% load('data1000D.mat');

% Choose d
% For the 2D dataset choose d = 1. For the 1000D dataset, you should choose d
% using the method indicated in part 1.2
% d = 1;
d = 30; % for data1000D.mat

% Prelims
D = size(X, 2);
n = size(X, 1);

% 1. PCA
[Z1, params1, Y1] = buggyPrinCompAnalysis(X, d);

% 2. Demeaned PCA
[Z2, params2, Y2] = deMeanPrinCompAnalysis(X, d);

% 3. Normalized PCA
% First demean and then normalize to have unit variance on each dimension
[Z3, params3, Y3] = normPrinCompAnalysis(X, d);

% 4. DRO
[Z4, params4, Y4] = DRO(X, d);

% 4. DRLV
[Z5, params5, Y5] = DRLV(X, d);

% Report reconstruction errors
err1 = sum(sum( (Y1-X).^2 ))/n;
err2 = sum(sum( (Y2-X).^2 ))/n;
err3 = sum(sum( (Y3-X).^2 ))/n;
err4 = sum(sum( (Y4-X).^2 ))/n;
err5 = sum(sum( (Y5-X).^2 ))/n;
fprintf('Reconstruction Errors:\n');
fprintf('Buggy PCA: %0.6f\n', err1);
fprintf('Demeaned PCA: %0.6f\n', err2);
fprintf('Normalized PCA: %0.6f\n', err3);
fprintf('DRO: %0.6f\n', err4);
fprintf('DRLV: %0.6f\n', err5);


% For 2D data, plot them out
if D ==2

  figure;
  plot(X(:,1), X(:,2), 'bo'); hold on
  plot(Y1(:,1), Y1(:,2), 'rx');
  axis([0 10 -5 13]);
  title('Buggy PCA');

  figure;
  plot(X(:,1), X(:,2), 'bo'); hold on
  plot(Y2(:,1), Y2(:,2), 'rx');
  axis([0 10 -5 13]);
  title('Demeaned PCA');

  figure;
  plot(X(:,1), X(:,2), 'bo'); hold on
  plot(Y3(:,1), Y3(:,2), 'rx');
  axis([0 10 -5 13]);
  title('Normalized PCA');

  figure;
  plot(X(:,1), X(:,2), 'bo'); hold on
  plot(Y4(:,1), Y4(:,2), 'rx');
  axis([0 10 -5 13]);
  title('DRO');

  figure;
  plot(X(:,1), X(:,2), 'bo'); hold on
  plot(Y5(:,1), Y5(:,2), 'rx');
  axis([0 10 -5 13]);
  title('DRLV');
end
