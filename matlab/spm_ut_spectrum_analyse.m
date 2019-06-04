% JS, Prague 2016
% Analyse H, c1, c2 with respect to the two database

clear all;
close all; clc;

sdata = 'train_1';
sdata = strcat(sdata, '_desc.mat');
path = '/home/jirka/data/kaggle_seisure_prediction_2016';

load(sdata);

ind = 1:size(data_desc,1);

aFiles = aFiles(ind);
data_desc = data_desc(ind,:);
y = data_desc(:,5);

fs = 400;

nLimSmallGaps_sec = 5;
%nLimSmallGaps_sec = 10^5;

%--- Estimation Parameters
% j1,j2 - scaling range
jmin = 1;
jmax = 10;

bPlot = 0;
bPlotSpec = 1;
Cum = 4; % number of cumulants
w1 = spectrum.welch;
w1.SegmentLength = 2^11;
overlap = round(w1.SegmentLength*0.5);%h.SegmentLength/2;

cnt = 0;
for kk = 21
    
    fprintf('%d/%d, loading file: %s\n', kk, length(aFiles), aFiles{kk});
    load(aFiles{kk});
    data_all = dataStruct.data;
    
    for iel = 1:size(data_all, 2)
        data = data_all(:,iel);
        
        data(data == 0) = NaN;
        
        %data = interpolateAllGaps(data,fs,1); % interpolate all gaps
        data = interpolateSmallGaps(data,fs,1,fs*nLimSmallGaps_sec);
        
        if bPlot
            xTime = (1/fs:1/fs:length(data)/fs)/60;
            h = figure;
            hold on;
            plot(xTime,data,'k')
            grid on;
            legend('FHR','Location','best')
        end
        
        %%%% spectrum - subtract the mean
        sig = data - nanmean(data);
        [P] = pwelch(sig,w1.SegmentLength,overlap,[],2,'onesided');
        ny = w1.SegmentLength;
        freq = [0:1:ny/2]/ny*fs ;
        
        if bPlotSpec
            figure(10001) %; clf
            hold on;
            plot(log2(freq),log2(P), 'k');
            grid on;
        end
        
%         pause
        %close all
        
        %return
        
        %%%% DWT
        gamint = 1;
        j1 = 6; j2 = 14;
        [cfeat,est,logstat] = featuresMF_RFL(data',j1,j2,gamint);
        index = find(logstat.q==2);
        Yjq = logstat.DWT.est(index,:);
        
        logstattemp=logstat.DWT;
        scales=log2(logstattemp.scale);
        
        pause
        
    end
    
    return
    
    %Yjq = Yjq(jmin:jmax);
    %scales = scales(jmin:jmax);
    
    cnt = cnt + 1;
    aaYjq_ldb(scales,cnt) = Yjq';
    H310_ldb(cnt,1) = cfeat.MF_HDWT;
    
    j1 = 4; j2 = 10;
    cfeat = featuresMF_RFL(logstat,j1,j2,gamint);
    H410_ldb(cnt,1) = cfeat.MF_HDWT;
    
    %scales = 3*fs_ldb/4 - 2.^scales;
    f_ldb = log2(3*fs_ldb/4) - scales;
    f_ldb = 2.^f_ldb;
    
    if bPlot
        figure(102); %clf;
        hold on;
        grid on;
        %plot(scales, Yjq, 'xk-');
        plot(log2(f_ldb), Yjq, 'xk-');
    end
    
    %%%% LWT
    gamint = 0.5;
    j1 = 3; j2 = 10;
    [cfeat,est,logstat] = featuresMF_RFL(data',j1,j2,gamint);
    
    scales=log2(logstat.LWT.scale);
    
    C1_j = logstat.LWT.est(end-Cum+1,:);
    C2_j = logstat.LWT.est(end-Cum+2,:);
    C3_j = logstat.LWT.est(end-Cum+3,:);
    
    aC1310_ldb(cnt,1) = cfeat.MF_c1;
    aC2310_ldb(cnt,1) = cfeat.MF_c2;
    aaC1_j_ldb(scales,cnt) = C1_j';
    aaC2_j_ldb(scales,cnt) = C2_j';
    
    j1 = 4; j2 = 10;
    [cfeat,est,logstat] = featuresMF_RFL(logstat,j1,j2,gamint);
    aC1410_ldb(cnt,1) = cfeat.MF_c1;
    aC2410_ldb(cnt,1) = cfeat.MF_c2;
    
    q = logstat.q;
    Did = (1:length (q)) + length (q);
    hid = (1:length (q)) + 2*length (q);
    
    h1 = est.LWT.t(hid) - gamint;
    D1 = est.LWT.t(Did);
    
    aah_ldb(:,cnt) = h1;
    aaD_ldb(:,cnt) = D1;
    
    if bPlot
        figure(62); %clf
        hold on
        plot(h1,D1,'xk-');
        %plot(h2,D2,'xr-');
        grid on;
        
        figure(103); %clf;
        hold on; grid on;
        %plot(log2(f_ldb), C1_j, 'xk-');
        plot(scales, C1_j, 'xk-');
        
        figure(104); %clf;
        hold on; grid on;
        %plot(log2(f_ldb), C2_j, 'xk-');
        plot(scales, C2_j, 'xk-');
    end
    %pause
    
end