close all; clear; clc; 
pkg load optim;

overall_time_start = time();

% --- DATASET 10 GENRES --- %
dataset_name = 'inputs/data.csv';
N = 10; % number of classes
% ------------------------- %

% --- DATASET 2 GENRES --- %
%dataset_name = 'inputs/data_2genre.csv';
%N = 2; % number of classes
% ------------------------ %

% --- VARIABLES --- %
perc_training = 0.75;
n_rep = 10;
C = [0.2:0.2:10]; % lambda for soft-margin

% reading the dateset from file
X = csvread(dataset_name);

% getting the size of the dataset
[rows, ~] = size(X);

% I ignore the first row because there are the labels of the columns
X = X(2:rows, :);

% I ignore the first column because it is the name of the file of the 
% audio sample and all values are different (as well they are not the
% titletracks)
% I ignore the last column because it is the label of the genre in a 
% string format; moreover I added a column with a numeric value for that
% field
X = X(:, 2:30);

% feature selection and reduction of the dataset
I = featureSelection(X, N);
%I = [2 1 4 5 6 7 29];  % features suggested by the maintainer of the dataset
X = X(:, I);

% split the dataset into two parts
[Xtr, Ytr, Xts, Yts] = splitDataset(X, perc_training, N);
[rowsXts, ~] = size(Xts);
[rowsYts, ~] = size(Yts);

% hold-out cross validation
lambda = holdoutCrossValidation(Xtr, Ytr, N, C, n_rep);

% parameter tuning
[classifiers, n] = OneVSOne(Xtr, Ytr, N, lambda);

% test
Ypred = zeros(rowsXts, 1);
yp_idx = 1;
for k = 1:rowsXts
    x = 1;
    pred = zeros(n, 1);
    for i = 1:N
      for j = i+1:N
          res = testSVM(classifiers(x).w, classifiers(x).b, Xts(k,:));
          if (res == 1)
              pred(x++) = i;
          else
              pred(x++) = j;
          endif
      endfor
    endfor
    Ypred(yp_idx++) = votingProtocol(pred);
endfor
overall_time_end = time();

err = calculateError(Yts, Ypred);
fprintf('Accuracy of test --> %d %%\n', (1 - err)*100);

minutes = round((overall_time_end - overall_time_start)/60);
seconds = mod(round(overall_time_end - overall_time_start), 60);
hours = floor(minutes/60);
minutes -= 60*hours;

fprintf('The test lasts for %d hours %d minutes and %d seconds\n', 
         hours, minutes, seconds);