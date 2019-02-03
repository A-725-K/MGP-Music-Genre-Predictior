%% This function select a subset of features of the dataset which are retained
%% important to predict the result.
%%
%% inputs:
%%  - X: dataset
%%  - N: number of classes

function I = featureSelection(X, N)
    [rows, cols] = size(X);
    step = rows/N;
    threshold = 0.01;
    
    S = zeros(N, cols-1);
    for j = 1:cols-1
        for i = 1:N
            c = (i-1)*step+1;
            mean_c = mean(X([c:c+step-1], j));
            mean_others = mean(X([1:c-1 c+step-1:end], j));
            var_c = var(X([c:c+step-1], j));
            var_others = var(X([1:c-1 c+step-1:end], j));
            S(i, j) = fisherCriterion(mean_c, mean_others, var_c, var_others);
        endfor
    endfor
    S_means = mean(S(:, :));
    
    I = selectColumns(S_means, threshold);
endfunction

%% Fisher's criterion to evaluate the goodness of a feature
function n = fisherCriterion(mean_c, mean_others, var_c, var_others)
    num = (mean_c - mean_others);
    den = sqrt(var_c + var_others);
    n = num/den;
endfunction

%% this function returns the index of the columns in the dataset
%% which seems to be more interesting than the others
function I = selectColumns(X, threshold)
    [~, cols] = size(X);
    I = [];
    index = 1;
    for i = 1:cols
        if (X(i) >= threshold)
            I(index) = i;
            index++;
        endif
    endfor
    I(index) = cols+1;
endfunction