%% This function split the dataset X into training and test sets
%%
%% inputs:
%%  - X: dataset to split
%%  - perc_tr: how much training set extract from the dataset (in percentage)
%%  - N: number of classes
%%
%% outputs:
%%  - Xtr, Ytr: training set
%%  - Xts, Yts: test set

function [Xtr, Ytr, Xts, Yts] = splitDataset(X, perc_tr, N)
    [rows, cols] = size(X);
    train_rows = floor(rows*perc_tr);
    test_rows = rows - train_rows;
    train_rows_per_class = train_rows/N;
    step = rows/N;
    
    Xtr = zeros(train_rows, cols - 1);
    Ytr = zeros(train_rows, 1);
    Xts = zeros(test_rows, cols - 1);
    Yts = zeros(test_rows, 1);

    xtr_index = 1;
    xts_index = 1;
    ytr_index = 1;
    yts_index = 1;
    
    for i = 1:step:rows
        I = randperm(step);
        X(i:i+step-1, :) = X(i + I(:) - 1, :); %row shuffle
        for j = 0:step-1
            if (i + j > rows)
                break
            endif
            if (j < train_rows_per_class)
                Xtr(xtr_index, :) = X(i + j, 1:(cols - 1));
                Ytr(ytr_index, :) = X(i + j, cols);
                xtr_index += 1;
                ytr_index += 1;
            else           
                Xts(xts_index, :) = X(i + j, 1:(cols - 1));
                Yts(yts_index, :) = X(i + j, cols);
                xts_index += 1;
                yts_index += 1;
            endif
        endfor
    endfor
endfunction
