close all; clear; clc; 
pkg load optim;

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
%C = [1:0.5:15]; % lambda for soft-margin
C = [1:0.2:4]; % lambda for soft-margin

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

%I = [2 1 4 5 6 7 29];
X = X(:, I);

% split the dataset into two parts
[Xtr, Ytr, Xts, Yts] = splitDataset(X, perc_training, N);
[rowsXts, ~] = size(Xts);
[rowsYts, ~] = size(Yts);


errs = zeros(29, 1);
errs_i = 1;
% get the model
for c = C
    [classifiers, n] = OneVSOne(Xtr, Ytr, N, c);

    Ypred = zeros(rowsYts, 1);
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

    %Ypred
    errs(errs_i) = calculateError(Yts, Ypred);
    fprintf('Accuracy -> %f %%\n', 1 - errs(errs_i++));
endfor

figure 831486;
plot(C, errs);
% cross-validation