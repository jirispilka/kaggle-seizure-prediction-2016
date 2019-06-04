% Test filtrace - dolni propoust, horni propust 

clear all;
close all; clc;

sdata = 'train_3';
sdata = strcat(sdata, '_desc.mat');
path = '/home/jirka/data/kaggle_seisure_prediction_2016';

load(sdata);
ind = 1:size(data_desc,1);
fs = 400;

bPlotSpec = 1;
w1 = spectrum.welch;
w1.SegmentLength = 2^11;
overlap = round(w1.SegmentLength*0.5);%h.SegmentLength/2;

bPlot = 0;
color = distinguishable_colors(16);

cnt = 0;

for kk = 583
    
    fprintf('%d/%d, loading file: %s\n', kk, length(aFiles), aFiles{kk});
    load(aFiles{kk});
    data_all = dataStruct.data;
    
    spm_ut_eegplot(data_all);
    
    %%
    for iel = 6%:size(data_all, 2)
        
        data = data_all(:,iel);
        data(data == 0) = NaN;
        
        data = interpolateAllGaps(data,fs,0); % interpolate all gaps
        data = removeNaNsAtBeginAndEnd(data);
              
        % find peaks
        t2 = (data);
        d = mapminmax(t2',-1,1)';
        d = d - mean(d);
        d = double([d;0]);
        d_filt = filtfilt((1/50)*ones(1,50),1,d.^2);
        
        peaks = peakdetect2(d_filt,15*median(d_filt));
        
        figure
        hold on;
        plot(data/8000, 'k')
        plot(d_filt, 'r')
        plot(1:100:length(d_filt),15*median(d_filt),'--c', 'linewidth',5)
        stem(peaks,-.1*ones(length(peaks),1),'.c');
        grid on;
        
        
        % high pass filter
        norder = 1;
        fc = .1;
        [b, a] = butter(norder,fc/(fs/2), 'high');
        data = filtfilt(b, a, double(data));
        
        figure
        freqz(b,1,fs);
              
        norder = 20;
        fc = 170;
        a = 1;
        b = fir1(norder,fc/(fs/2), 'low');
        
        figure
        freqz(b,1,fs);
        
        y = filtfilt(b, a, double(data));
        
        figure
        plot(data,'b'); hold on;
        plot(y,'r','linewidth',1);
        hold on;
        plot(y-data, 'k')
                    
        %%%% spectrum - subtract the mean
        sig = data - nanmean(data);
        [P] = pwelch(sig,w1.SegmentLength,overlap,[],2,'onesided');
        ny = w1.SegmentLength;
        freq = [0:1:ny/2]/ny*fs ;
        
        P2 = pwelch(y,w1.SegmentLength,overlap,[],2,'onesided');
        
        if bPlotSpec
            figure(10001) %; clf
            hold on;
            plot((freq),log2(P), 'color', color(iel,:));
            plot((freq),log2(P2), 'color', 'r');
            grid on;
        end
    end
end
