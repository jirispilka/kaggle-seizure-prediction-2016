function cfeat = spm_ut_compute_feat_multivariate(data, sel_feat, fs)

cfeat = [];

bMakeAllNaN = false;
if isnan(data),
    % compute with random and make all NaN at the end
    data = randn(size(data));
    bMakeAllNaN = true;
end

% data = removeNaNsAtBeginAndEnd(data);

%% statistics
if findStr(sel_feat,'corr') %|| findStr(sSelFeat,'all')
    
    %idx = ~sum(isnan(data),2);
    %A = corrcoef(data(idx, :));
    %B = triu(A,1);
    
    for i = 1:size(data, 2)
        for j = 1:size(data, 2)
            if j > i
                idx = ~sum(isnan(data(:,[i,j])),2);
                A = corrcoef(data(idx, [i,j]));
                B = A(1,2);
                name = sprintf('corr_%d%d',i,j);
                cfeat.(name) = B;
            end
        end
    end
    
    idx = ~sum(isnan(data),2);
    X = data(idx, :);
    eig_max = 16;
    if ~isempty(X)
        A = corrcoef(X);
        D = eig(A);
        D = sort(D, 'descend');
    else
        D = nan(eig_max,1);
    end

    for i = 1:eig_max
        name = sprintf('eigenvalue_%d',i);
        cfeat.(name) = D(i);
    end
    
    X = data';
    idx = ~sum(isnan(data),2);
    mult = 1;
    ds = ones(size(X,1),1);
    co = ASpearman1_initialization(mult);
    r = ASpearman1_estimation(X(:,idx),ds,co);
    cfeat.spearman_mult = r;
        
end

%% MAKE ALL FEATURES NAN
if bMakeAllNaN
    cnames = fieldnames(cfeat);
    for i = 1:length(cnames)
        cfeat.(cnames{i}) = NaN;
    end
end

function b = findStr(str,sFind)

str = lower(str);
sFind = lower(sFind);

if isempty(strfind(str,sFind))
    b = false;
else
    b = true;
end
