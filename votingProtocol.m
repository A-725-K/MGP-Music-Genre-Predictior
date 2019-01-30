%% This function compute to which class a point belongs
%% selecting the class that receives the majority of
%% votes given by the classifiers.
%% In case of draw the first class that receives the majority
%% of votes is chosen
%% 
%% input:
%%  - X: set of votes
%%
%% output:
%%  - res: the class which receives most of the votes

function res = votingProtocol(X)
    if (~ isvector(X))
        error('X must be a vector');
    endif
    
    [rows, ~] = size(X);
    X = sort(X);
    max_count = 1;
    res = X(1);
    current_count = 1;
    
    for i = 2:rows
        if (X(i) == X(i - 1))
            current_count++;
        else
            if (current_count > max_count)
                max_count = current_count;
                res = X(i - 1);
            endif
            current_count = 1;
        endif
    endfor
    
    % check for last element
    if (current_count > max_count)
        max_count = current_count;
        res = X(rows);
    endif
endfunction
