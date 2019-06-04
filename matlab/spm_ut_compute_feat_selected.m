function cfeat = spm_ut_compute_feat_selected(data,sel_feat, fs)

cfeat = [];

bMakeAllNaN = false;
if isnan(data),
    % compute with random and make all NaN at the end
    data = randn(size(data));
    bMakeAllNaN = true;
end

%aDataPossibleNaN = data;
data = removeNaNsAtBeginAndEnd(data);

data = data - nanmean(data);

%% statistics
if findStr(sel_feat,'stat') %|| findStr(sSelFeat,'all')
    
    idx = ~isnan(data);
    %cfeat.mean = mean(data(idx));
    cfeat.std = std(data(idx));
    %cfeat.median = median(data(idx));
    %cfeat.mad = mad(data(idx));
    cfeat.skewness = skewness(data(idx), 0);
    cfeat.kurtosis = kurtosis(data(idx), 0);
    %[~,~,kstat] = lillietest(data(idx));
    %cfeat.kstat = kstat;
    d = diff(data(idx));
    d_std = std(d);
    cfeat.hjort2 = d_std/cfeat.std;
    cfeat.hjort3 = std(diff(d))/d_std/cfeat.hjort2;
end

%% spectral
if findStr(sel_feat,'spectral')% || findStr(sel_feat,'all')
    
    % interpolate all
    temp_data = interpolateAllGaps(data, fs, 0);
    temp_data = removeNaNsAtBeginAndEnd(temp_data);
    
    [tot_pow, pow_split, pow_split2Hz, alpha] = spm_features_spectral(temp_data,fs,[],0);
    cfeat.e_delta = pow_split(1);
    cfeat.e_theta = pow_split(2);
    cfeat.e_alpha = pow_split(3);
    cfeat.e_beta = pow_split(4);
    cfeat.e_low_gamma = pow_split(5);
    cfeat.e_high_gamma = pow_split(6);
    cfeat.energy_tot = tot_pow;
    cfeat.spectrum_slope = alpha;
    
    cfeat.energy_0002= pow_split2Hz(1);
    cfeat.energy_0204 = pow_split2Hz(2);
    cfeat.energy_0406 = pow_split2Hz(3);
    cfeat.energy_0608 = pow_split2Hz(4);
    cfeat.energy_0810 = pow_split2Hz(5);
    cfeat.energy_1012 = pow_split2Hz(6);
    cfeat.energy_1218 = pow_split2Hz(7);
    cfeat.energy_1824 = pow_split2Hz(8);
    cfeat.energy_2430 = pow_split2Hz(9);
    cfeat.energy_3040 = pow_split2Hz(10);
    cfeat.energy_4050 = pow_split2Hz(11);
    cfeat.energy_5060 = pow_split2Hz(12);

end

%% spectralentropy
if findStr(sel_feat,'sp_entropy')% || findStr(sel_feat,'all')
    
    % interpolate all
    temp_data = interpolateAllGaps(data, fs, 0);
    temp_data = removeNaNsAtBeginAndEnd(temp_data);
    
    cfeat = spm_features_spectral_entropy(temp_data,fs,[]);
end

%% H, c1, c2 as function of j

if findStr(sel_feat,'mfj') || findStr(sel_feat,'all')
    gamint = 1;
    j1 = 2;
    j2 = 7;
    [cf,est,logstat] = featuresMF_RFL(data',j1,j2,gamint);
    cfeat.H27 = cf.MF_HDWT;
    cfeat.c1 = cf.MF_c1;
    cfeat.c2 = cf.MF_c2;
    cfeat.c3 = cf.MF_c3;
    cfeat.c4 = cf.MF_c4;
    %cFeatures.hmin = cfeat.MF_hmin_noint;
    
    jmax = 11;
    % save c1 and c2 as function of j
    for j = 1:jmax
        name = strcat('H_j_',num2str(j));
        if j > length(cf.H_j)
            cfeat.(name) = NaN;
        else
            cfeat.(name) = cf.H_j(j);
        end
    end
    
    for j = 2:jmax
        name = strcat('c1_j_',num2str(j));
        if j > length(cf.c1_j)
            cfeat.(name) = NaN;
        else
            cfeat.(name) = cf.c1_j(j);
        end
    end
    
    for j = 2:jmax
        name = strcat('c2_j_',num2str(j));
        if j > length(cf.c2_j)
            cfeat.(name) = NaN;
        else
            cfeat.(name) = cf.c2_j(j);
        end
    end
    
    for j = 2:jmax
        name = strcat('c3_j_',num2str(j));
        if j > length(cf.c3_j)
            cfeat.(name) = NaN;
        else
            cfeat.(name) = cf.c3_j(j);
        end
    end
    
    for j = 2:jmax
        name = strcat('c4_j_',num2str(j));
        if j > length(cf.c4_j)
            cfeat.(name) = NaN;
        else
            cfeat.(name) = cf.c4_j(j);
        end
    end
end

%% entropywav
if findStr(sel_feat,'wav_entropy') || findStr(sel_feat,'all')
    cTemp = spm_features_wavelet_entropy(data);
    names = fieldnames(cTemp);
    for i = 1:length(names)
        for j = 1:length(cTemp.(names{i}))
            name = strcat(names{i},num2str(j));
            cfeat.(name) = cTemp.(names{i})(j);
        end
    end
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
