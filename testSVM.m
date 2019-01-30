%% classify a point with SVM parameters
%% 
%% inputs:
%%  - w: slope
%%  - b: intercept
%%  - x: new point that needs a classification
%%
%% output:
%%  - val: a value which could be +1 or -1 that identifies
%%         the class of the new point x

function val = testSVM(w, b, x)
    val = sign(x*w + b);
endfunction
