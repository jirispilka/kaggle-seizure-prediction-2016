% JS, Prague 2016
% Misc analysis

clear all;
close all; clc;

sdata = 'train_1';
%sdata = 'test_3';
sdata = strcat(sdata, '_desc.mat');
path = '/home/jirka/data/kaggle_seisure_prediction_2016';

load(sdata);

ind = 1:size(data_desc,1);

fs = 400;

%nLimSmallGaps_sec = 5;
%nLimSmallGaps_sec = 10^5;

bPlotSpec = 1;
w1 = spectrum.welch;
w1.SegmentLength = 2^11;
overlap = round(w1.SegmentLength*0.5);%h.SegmentLength/2;

bPlot = 0;
color = distinguishable_colors(16);

cnt = 0;

for kk = 2
    
    fprintf('%d/%d, loading file: %s\n', kk, length(aFiles), aFiles{kk});
    load(aFiles{kk});
    data_all = dataStruct.data;
    
    spm_ut_eegplot(data_all);
%     x = data_all(16800+340:16800+400,7)
    
    %%
    figure
    spectrogram(double(data_all(:, 8)), 2^12)
    
    %figure
    %spectrogram(double(data_all(:, 1)), 8192)
    
    %figure
    %plot(data_all(:,1), data_all(:,3), 'xk')
    %grid on;
    
%     corr(data_all(:, 1), data_all(:, 3))
%     corr(data_all(:, 1), data_all(:, 3), 'type', 'Spearman')
    
    %%    
    for iel = 1:size(data_all, 2)
        
        data = data_all(:,iel);
        data(data == 0) = NaN;
        
        data = interpolateAllGaps(data,fs,0); % interpolate all gaps
        data = removeNaNsAtBeginAndEnd(data);
        
%         if bPlot
%             xTime = (1/fs:1/fs:length(data)/fs)/60;
%             h = figure;
%             hold on;
%             plot(xTime,data,'color', color(iel,:))
%             grid on;
%             legend('FHR','Location','best')
%         end
        %spm_features_spectral(data,fs,[],0)
        
        %%%% spectrum - subtract the mean
        sig = data - nanmean(data);
        [P] = pwelch(sig,w1.SegmentLength,overlap,[],2,'onesided');
        ny = w1.SegmentLength;
        freq = [0:1:ny/2]/ny*fs ;
        
        if bPlotSpec
            figure(10001) %; clf
            hold on;
            plot((freq),log2(P), 'color', color(iel,:));
            grid on;
        end
        
        %[tot_pow,pow_split,alpha] = spm_features_spectral_entropy(X,fs,h,verbose)
        
%         m = 3;
%         r = 0.2;
%         knn = 10;
%         norm_data = data/nanstd(data);
%         
%         [apen,samp,entropyANN] = ComputeEntropyAll(norm_data',m,r,knn,fs);
%         %cFeatures.ApEn302 = apen;
%         cFeatures.SampEn302 = samp;
%         cFeatures.EntropyANN = entropyANN;

        
        
    end
    
end
