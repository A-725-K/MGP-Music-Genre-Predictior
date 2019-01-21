function err = calculateError(Yts, Ypred)
    if (~ isvector(Yts) || ~ isvector(Ypred))
        error('Yts and Ypred must be vectors');
    endif
    [rows, ~] = size(Yts);
    Yts(Yts ~= 1) = -1;
    err = 0;
    for i = 1:rows
        if (Ypred(i) ~= Yts(i))
            err++;
        endif
    endfor
    err /= rows;
endfunction
