close all; clear; clc; 
pkg load optim;

% --- DATASETS --- %
dataset_name = 'inputs/data.csv';
N = 10; % number of classes
%%dataset_name = 'inputs/data_2genre.csv';
%%N = 2; % number of classes

% --- VARIABLES --- %
perc_training = 0.75;
C = 1; %lambda for soft-margin

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

%I = [2 1 4 5 6 7];
X = X(:, I);

% split the dataset into two parts
[Xtr, Ytr, Xts, Yts] = splitDataset(X, perc_training, N);

% get the model
%[classifiers, n] = OneVSOne(Xtr, Ytr, N, C);
load('classifiers');

%for i = size(Xts)(1)
   x = 1;
   pred = zeros(45, 1);
   for i = 1:10
       for j = i+1:10
            res = testSVM(classifiers(x).w, classifiers(x).b, Xts(35,:));
            if (res == 1)
                pred(x++) = i;
            else
                pred(x++) = j;
            endif
        endfor
    endfor
val = votingProtocol(pred)
%endfor

% cross-validation


