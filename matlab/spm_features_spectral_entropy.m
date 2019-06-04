function [cfeat] = spm_features_spectral_entropy(X,fs,h)
% Routine for doing frequency analysis

% Instantiate spectrum object and call its PSD method.
if nargin < 3 || isempty(h)
    h = spectrum.welch;
    h.SegmentLength = 2^12;
end

% if nargin < 4
%     verbose = 0;
% end

% subtract the mean
X = X - mean(X);

% JS: 2013-11-05 - compute PSD using welch method, the psd was not working (Matlab 2012a) 
win = h.SegmentLength;
overlap = round(h.SegmentLength*0.5);%h.SegmentLength/2;
[P,f] = pwelch(X,win,overlap,[],fs,'onesided');
P = double(P);

delta = inband_entropy(f,P,0.1,4-eps); 
theta = inband_entropy(f,P,4,8-eps);
alpha = inband_entropy(f,P,8,12-eps);
beta = inband_entropy(f,P,12,30-eps);
low_gamma = inband_entropy(f,P,30,70-eps);
high_gamma = inband_entropy(f,P,70,180);
tot = inband_entropy(f,P,0.1,180);

cfeat.delta_H_sh = delta(1);
cfeat.delta_H_sh_knn = delta(2);
cfeat.delta_H_sh_ann = delta(3);

cfeat.theta_H_sh = theta(1);
cfeat.theta_H_sh_knn = theta(2);
cfeat.theta_H_sh_ann = theta(3);

cfeat.alpha_H_sh = alpha(1);
cfeat.alpha_H_sh_knn = alpha(2);
cfeat.alpha_H_sh_ann = alpha(3);

cfeat.beta_H_sh = beta(1);
cfeat.beta_H_sh_knn = beta(2);
cfeat.beta_H_sh_ann = beta(3);

cfeat.low_gamma_H_sh = low_gamma(1);
cfeat.low_gamma_H_sh_knn = low_gamma(2);
cfeat.low_gamma_H_sh_ann = low_gamma(3);

cfeat.high_gamma_H_sh = high_gamma(1);
cfeat.high_gamma_H_sh_knn = high_gamma(2);
cfeat.high_gamma_H_sh_ann = high_gamma(3);

cfeat.tot_H_sh = tot(1);
cfeat.tot_H_sh_knn = tot(2);
cfeat.tot_H_sh_ann = tot(3);

%% compute band energy
function [H] = inband_entropy(f, Pxx, fl, fh)

ind = (fl <= f & f <= fh);

P = Pxx(ind);

% if length(P) > 25
%     P = log(P);
% end

% P = log(P);
% P = P + abs(min(P)) + eps;
P = P/sum(P);

mult = 1;
co = HShannon_kNN_k_initialization(mult); %initialize the entropy (’H’) estimator

H_sh = sum(P.*log(P));
H_sh_kNN = HShannon_kNN_k_estimation(P',co);       

% P = Pxx(ind);
% P = P/std(P);
m = 2;
k_nn = 3;
H_ANN = compute_entropy(P,k_nn,m+1,1) - compute_entropy(P,k_nn,m,1);

H = [-H_sh, -H_sh_kNN, -H_ANN];

