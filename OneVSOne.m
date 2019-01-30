%% One-Vs-One approach: this algorithm creates a list of
%% N(N-1)/2 classifiers, where N is the number of classes.
%% Those are built with SVM as a learner. Each classifier
%% is a struct composed by two fields w and b which are 
%% the parameters of the loss function f(x) = xw + b
%%
%% inputs:
%%  - Xtr: training set
%%  - Ytr: labels for supervised training
%%  - N: number of classes
%%  - C: lambda for soft margin
%% 
%% outputs:
%%  - classifiers: list of trained classifiers
%%  - nr_classifiers: the number of classifiers obtained
    
function [classifiers, nr_classifiers] = OneVSOne(Xtr, Ytr, N,C)  
    [rows, ~] = size(Xtr);
    nr_classifiers = N*(N-1)/2;
    el_per_class = floor(rows / N);
    
    % 1-2, 1-3, 1-4, 1-5, 1-6, 1-7, 1-8, 1-9, 1-10
    % 2-3, 2-4, 2-5, 2-6, 2-7, 2-8, 2-9, 2-10
    % 3-4, 3-5, 3-6, 3-7, 3-8, 3-9, 3-10
    % 4-5, 4-6, 4-7, 4-8, 4-9, 4-10
    % 5-6, 5-7, 5-8, 5-9, 5-10
    % 6-7, 6-8, 6-9, 6-10
    % 7-8, 7-9, 7-10
    % 8-9, 8-10
    % 9-10
    idx_class = 1;
    for fst_class = 1:N
        for snd_class = fst_class+1:N
            fst_1 = (fst_class-1)*el_per_class+1;
            lst_1 = fst_class*el_per_class;
            fst_2 = (snd_class-1)*el_per_class+1;
            lst_2 = snd_class*el_per_class;
            [w, b] = SVM(Xtr([fst_1:lst_1 fst_2:lst_2], :), ...
                         Ytr([fst_1:lst_1 fst_2:lst_2]), C);
            classifiers(idx_class).w = w;
            classifiers(idx_class).b = b;
            idx_class++;
        endfor
    endfor
endfunction