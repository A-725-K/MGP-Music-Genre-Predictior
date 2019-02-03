%% Support Vector Machine for classification
%%
%% inputs:
%%  - X: dataset
%%  - Y: labels
%%  - C: lambda for soft-margin SVM
%%
%% outputs: model
%%  - w: slope 
%%  - b: intercept

function [w, b] = SVM(X, Y, C) 
    if (~ isvector(Y))
        error('Y must be a vector');
    endif
    
    [rows, ~] = size(X);
    el_per_class = floor(rows / 2);
    n_iter = 600;  % iterations allowed in qp
    threshold = 0.0001; % tau for finding support vectors
    
    % split in half the training set labels
    % the only permitted labels are {+1, -1}
    Y(1:el_per_class) = 1;
    Y(el_per_class+1:end) = -1;
    
    % parameters for quadratic programming solver (qp)
    X0 = zeros(rows, 1);
    H = (X*X') .* (Y*Y');
    F = -ones(rows, 1);
    A = [];
    B = [];
    AEQ = Y';
    BEQ = 0;
    LB = zeros(rows, 1);
    UB = C*ones(rows, 1);
    options = optimset('MaxIter', n_iter);
                                
    [Alpha, ~, info, ~] = qp(X0, H, F, A, B, LB, UB, options);
    
    SV = findSupportVectors(Alpha, 0, C, threshold);
     if (~ isvector(SV) || size(SV, 1) == 0)
        fprintf('Did not found any support vector\n');
        w = 0;
        b = 0;
        return;
    endif
    w = computeW(X, Y, Alpha, SV);
    b = calculateB(X, Y, Alpha, SV);
endfunction

% compute the vector w (slope)
function w = computeW(X, Y, Alpha, SV)
     if (~ isvector(Alpha) || ~ isvector(Y))
        Alpha
        error('Alpha and Y must be1 vectors');
    endif
    [~, colsX] = size(X);
    w = zeros(1, colsX);
    for i = SV'
        w += Alpha(i) * Y(i) * X(i, :);
    endfor
    w = w';
endfunction

% find the support vectors, those which in X are nonnegative
function SV = findSupportVectors(Alpha, lb, ub, threshold)
    if (~ isvector(Alpha))
        error('Alpha must be a vector');
    endif
    SV = [];
    index = 1;
    [rows, ~] = size(Alpha);
    for i = 1:rows
        if (Alpha(i) >= threshold && Alpha(i) >= lb && Alpha(i) <= ub)
            SV(index++, 1) = i;
        endif
    endfor
endfunction

% compute the value of b (intercept)
function b = calculateB(X, Y, Alpha, SV)
    if (~ isvector(Alpha) || ~ isvector(Y))
        error('Alpha and Y must be vectors');
    endif
    [Ns, ~] = size(SV);
    b = 0;
    for s = SV'
        sum = 0;       
        for m = SV'
            sum += (X(m, :) * X(s, :)') * Alpha(m) * Y(m);
        endfor
        b += (Y(s) - sum);
    endfor
    b = b/Ns;
endfunction