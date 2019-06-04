function cFeat = spm_features_wavelet_entropy(X)
% For given time series compute entropy on Dx, Ax, Lx
%
% Input:
%  data - [nx1] file name
%  m - [int] embedding dimension
%  r - [int] tolerance parameter
%
% Output:
%  cFeat - [struct] contains the computed entropies
%
% Jiri Spilka, Patrice Abry, ENS Lyon 2014

Nwt=3;

X = double(X);

[mcol mrow] = size(X);
if mcol > mrow
    X = X';
end

[coefB, leadersB, nj] =  DxLx1d_js(X, Nwt);
nscale=length(coefB);

jmax = 11;
nUsedScales = min(jmax, nscale);
%nUsedScales = nscale;

z = nan(1, jmax);
EnDxKnn = z; EnAxKnn = z;
EnDxN = z; EnAxN = z; EnLxN = z;

for ia=2:nUsedScales % loop on the scales
    
    tmpDx = coefB(ia).value_noabs;
    tmpAx = coefB(ia).approW;
    tmpLx = leadersB(ia).value;
    
    % % entropy of raw DWT coefs
    %EnDxKnn(ia) = en_ite_knn(tmpDx);
    %EnAxKnn(ia) = en_ite_knn(tmpAx);
    
    % % Entropy of absolute value of DWT coefs
    %[ApTDB_Abs(ia), SeTDB_Abs(ia), EnTDB_Abs(ia)] = ComputeEntropyAll(abs(tmpB),m,r);
    
    % entropy of normalized coefs
    if sum(isnan(tmpDx)) == length(tmpDx) || length(tmpDx) == 1
        EnDxN(ia) = NaN;
    else
        temp=tmpDx./nanstd(tmpDx);
        EnDxN(ia) = en_stephane_ann(temp);
    end
    
    if sum(isnan(tmpAx)) == length(tmpAx) || length(tmpAx) == 1
        EnAxN(ia) = NaN;
    else
        temp=tmpAx./nanstd(tmpAx);
        EnAxN(ia) = en_stephane_ann(temp);
    end
    
%     if sum(isnan(tmpLx)) == length(tmpLx) || length(tmpLx) == 1
%         EnAxN(ia) = NaN;
%     else
%         temp=tmpLx./nanstd(tmpLx);
%         EnLxN(ia) = en_stephane_ann(temp);
%     end
    
    % entropy of the log absolute value of DWT coefs
    %     temp=log(abs(tmpDx));
    %     temp=temp/std(temp);
    %     [ApLogAbsDx(ia),SeLogAbsDx(ia),EnLogAbsDx(ia)] = ComputeEntropyAll(temp,m,r);
    %
    %     temp=log(abs(tmpAx));
    %     temp=temp/std(temp);
    %     [ApLogAbsAx(ia),SeLogAbsAx(ia),EnLogAbsAx(ia)] = ComputeEntropyAll(temp,m,r);
    %
    %     temp=log(abs(tmpLx));
    %     temp=temp/std(temp);
    %     [ApLogAbsLx(ia),SeLogAbsLx(ia),EnLogAbsLx(ia)] = ComputeEntropyAll(temp,m,r);
    
    % %     % Entropy of normalized absolute value of DWT coefs
    % %     temp=abs(tmpB)/std(abs(tmpB));
    % %     [ApTDBN_Abs(ia),SeTDBN_Abs(ia),EnTDBN_Abs(ia)]= ComputeEntropyAll(temp,m,r);
    % %
    % %     % entropy of the log of  DWT coefs
    % %     temp=log(abs(tmpB));
    % %     [ApTDB_logAbs(ia),SeTDB_logAbs(ia),EnTDB_logAbs(ia)]= ComputeEntropyAll(temp,m,r);
    % %
    % %     % entropy of the normalized log of  DWT coefs
    % %     temp=temp/std(temp);
    % %     [ApTDB_logAbsN(ia),SeTDB_logAbsN(ia),EnTDB_logAbsN(ia)]= ComputeEntropyAll(temp,m,r);
end

%% save all
% cFeat.EnDxKnn = EnDxKnn;
% cFeat.EnAxKnn = EnAxKnn;

cFeat.EnDxN = EnDxN;
cFeat.EnAxN = EnAxN;
% cFeat.EnLxN = EnLxN;

function H = en_ite_knn(X)

[ncol,nrow] = size(X);
if ncol > nrow
    X = X';
end

% if sum(sum(isnan(X))) > 0
%     a = 5;
% end

X = X(~isnan(X));

if isempty(X) || length(X) <= 10
    H = NaN;
    return
end

X = X/sum(X);

mult = 1;
co = HShannon_kNN_k_initialization(mult); %initialize the entropy (’H’) estimator
H = -HShannon_kNN_k_estimation(X,co);

if isinf(H)
    H = NaN;
end

%%
function H = en_stephane_ann(X)

k_nn = 10;
m = 3;

labels = ones(size(X));
labels(isnan(X) | isinf(X)) = 0;
X(labels == 0) = 0;
em3 = compute_entropy(X,k_nn,m,1,labels);
em2 = compute_entropy(X,k_nn,m-1,1,labels);
H =  em3 - em2;

if H == 0
    H = NaN;
end