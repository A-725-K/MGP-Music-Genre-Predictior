% Support Vector Machine for classification
function val = SVM(X, Y, new_x) 
    if (~ isvector(Y))
        error('Y must be a vector');
    endif
    
    [rows, ~] = size(X);
    threshold = 0.0001;
    
    Y(Y ~= 1) = -1; % the only permitted labels are {+1, -1}
    
    % parameters for quadratic programming solver
    H = (X*X') .* (Y*Y');
    F = -1*ones(rows, 1);
    A = [];
    B = [];
    AEQ = Y';
    BEQ = 0;
    LB = zeros(rows, 1);
    UB = ones(rows, 1);
    
    [Alpha, ~, EXITFLAG, ~, ~] = quadprog(H, F, A, B, AEQ, BEQ, LB, UB);
    if (EXITFLAG == 0)
        error('This iteration does not converge');
    endif
    
    SV = findSupportVector(Alpha, threshold);
    w = computeW(X, Y, Alpha, SV);
    b = calculateB(X, Y, Alpha, SV);
    
    %classify new point
    val = sign(new_x*w + b);
endfunction

% compute the vector w (slope)
function w = computeW(X, Y, Alpha, SV)
    if (~ isvector(Alpha) || ~ isvector(Y) || ~ isvector(SV))
        error('Alpha,Y and SV must be vectors');
    endif
    [rowsAlpha, ~] = size(Alpha);
    [~, colsX] = size(X);
    w = zeros(1, colsX);
    for i = SV'
        w += Alpha(i) * Y(i) * X(i, :);
    endfor
    w = w';
endfunction

% find the support vectors, those which in X are nonnegative
function SV = findSupportVector(Alpha, threshold)
    if (~ isvector(Alpha))
        error('Alpha must be a vector');
    endif
    SV = [];
    index = 1;
    [rows, ~] = size(Alpha);
    for i = 1:rows
        if (Alpha(i) >= threshold)
            SV(index++, 1) = i;
        endif
    endfor
endfunction

% compute the vector b (intercept)
function b = calculateB(X, Y, Alpha, SV)
    if (~ isvector(Alpha) || ~ isvector(Y) || ~ isvector(SV))
        error('Alpha,Y and SV must be vectors');
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