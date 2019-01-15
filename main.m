close all; clear; clc; 

%% --- VARIABLES --- %%
dataset_name = 'inputs/data.csv';
perc_training = 0.75;
N = 10; % number of classes

% reading the dateset from file
X = csvread(dataset_name);

[rows, cols] = size(X);

% I ignore the first row because there are the labels of the columns
X = X(2:rows, :);

% I ignore the first column because it is the name of the file of the 
% audio sample and all values are different (as well they are not the
% titletracks)
% I ignore the last column because it is the label of the genre in a 
% string format; moreover I added a column with a numeric value for that
% field
X = X(:, 2:30);

% split the dataset into two parts
[Xtr, Ytr, Xts, Yts] = splitDataset(X, perc_training, N);

% variable selection
% ONEvsALL
% SVM algorithm
% cross-validation
% training
% test



