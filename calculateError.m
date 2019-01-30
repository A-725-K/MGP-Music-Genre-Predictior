%% This function compute the error of the algorithm as the
%% number of wrong predictions on number of samples
%%
%% inputs:
%%  - Yts: correct labels
%%  - Ypred: predictions of my algorithm
%%
%% output:
%%  - err: percentage of error

function err = calculateError(Yts, Ypred)
    if (~ isvector(Yts) || ~ isvector(Ypred))
        error('Yts and Ypred must be vectors');
    endif
    [rows, ~] = size(Yts);
    
    err = 0;
    for i = 1:rows
        if (Ypred(i) ~= Yts(i))
            err++;
        endif
    endfor
    err /= rows;
endfunction
