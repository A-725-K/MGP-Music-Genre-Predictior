function lambda = holdoutCrossValidation(Xtr, Ytr, N, C, n_iter)
    [~, colsC] = size(C);
    idx_accs = 1;
    idx_times = 1;
    
    % 2/3 of the training set is used for training
    % 1/3 of the training set is used for validation
    perc_tr = 0.666; 
    
    % rebuilt the dataset
    X = Xtr;
    X(:, end + 1) = Ytr;
    
    % accuracy of n_iter tests each c
    c_accs = zeros(n_iter, colsC);
    % times of n_iter tests each c 
    c_times = zeros(n_iter, colsC);

    for c = C
        fprintf('C = %0.1f\n', c);
        for ni = 1:n_iter
            [Xtr_tr, Ytr_tr, Xtr_ts, Ytr_ts] = splitDataset(X, perc_tr, N);
            [rowsXtr_ts, ~] = size(Xtr_ts);
            [rowsYtr_ts, ~] = size(Ytr_ts);
            
            errs = zeros(colsC, 1);
            idx = 1;
            
            time_before = time();
            
            [classifiers, n] = OneVSOne(Xtr_tr, Ytr_tr, N, c);
            if (n == 0)
                continue;
            endif
            Ypred = zeros(rowsYtr_ts, 1);
            yp_idx = 1;
            for k = 1:rowsXtr_ts
                x = 1;
                pred = zeros(n, 1);
                for i = 1:N
                    for j = i+1:N
                        res = testSVM(classifiers(x).w, classifiers(x).b, Xtr_ts(k,:));
                        if (res == 1)
                            pred(x++) = i;
                        else
                            pred(x++) = j;
                        endif
                    endfor
                endfor
                Ypred(yp_idx++) = votingProtocol(pred);
            endfor
            time_after = time();
            errs(idx) = calculateError(Ytr_ts, Ypred);
            c_times(ni, idx_times) = time_after - time_before;
            c_accs(ni, idx_accs) = (1 - calculateError(Ytr_ts, Ypred))*100;
            fprintf('%d) Accuracy -> %0.2f %%\n', ni, (1 - errs(idx++))*100);
        endfor
        idx_accs++;
        idx_times++;
    endfor

    figure('Name', 'Accuracy on average', 'NumberTitle', 'off');
    plot(C, mean(c_accs), '-*g', 'LineWidth', 2);
    xlabel('lambda');
    ylabel('accuracy (%)');
    grid on;
    
    figure('Name', 'Performance', 'NumberTitle', 'off');
    plot(C, mean(c_times), '-dr', 'LineWidth', 3);
    xlabel('lambda');
    ylabel('time (s)');
    grid on;
    
    [max_mean, max_idx] = max(mean(c_accs));
    lambda = C(:, max_idx);
    fprintf('The best C was %0.1f with an average accuracy of %0.2f %%\n', 
            lambda, max_mean);
endfunction
