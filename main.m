close all; clear; clc; 
pkg load optim;

%% --- VARIABLES --- %%
%dataset_name = 'inputs/data.csv';
dataset_name = 'inputs/data_2genre.csv';
%N = 10; % number of classes
N = 2; % number of classes
perc_training = 0.75;

% reading the dateset from file
X = csvread(dataset_name);

[rows, ~] = size(X);

% I ignore the first row because there are the labels of the columns
X = X(2:rows, :);

% I ignore the first column because it is the name of the file of the 
% audio sample and all values are different (as well they are not the
% titletracks)
% I ignore the last column because it is the label of the genre in a 
% string format; moreover I added a column with a numeric value for that
% fieldw
X = X(:, 2:30);

% feature selection
I = featureSelection(X, N);

%I = [2 1 4 5 6 7];
X = X(:, I);

% split the dataset into two parts
[Xtr, Ytr, Xts, Yts] = splitDataset(X, perc_training, N);

%Xtr = Xtr(1:150, :);
%Ytr = Ytr(1:150);

v = SVM(Xtr, Ytr, Xts(1, :));

% -------------------- %
% ------- TEST ------- %
% -------------------- %
%%[rowsYts, ~] = size(Yts);
%%index = 0;
%%Ypred = zeros(rowsYts, 1);
%%for i = 1:50
%%    Ypred(i, 1) = SVM(Xtr, Ytr, Xts(i, :));
%%endfor
%%calculateError(Yts, Ypred)

% OneVSAll
% cross-validation


