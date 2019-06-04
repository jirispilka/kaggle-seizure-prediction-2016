clear all;
close all;
clc;

data_folder = 'train_3';
N = 240000;

nfrom = 582;
nto = 584;

sdata = strcat(data_folder, '_desc.mat');
load(sdata);

channel = 16;
data_all = [];
for kk = nfrom:nto

        
    fprintf('loading file: %s\n', aFiles{kk});
    
    d = load(aFiles{kk});
    data_all = [data_all; d.dataStruct.data(:, [channel])];
    
    
%     x = data_all(16800+340:16800+400,7)
    
    
end

%%
spm_ut_eegplot(data_all);
figure(1435)
a = axis;
cnt = 0;
for kk = nfrom:nto
    cnt = cnt + 1;
    plot([cnt*N cnt*N], [a(3) a(4)], '--k');
end

%%

% figure
% spectrogram(double(data_all(:, 1)), 2^11)

fs = 400;
N = 240000;

win_min = 1; % window length for analysis
win_step_min = 0.5; % time step
win_samp = win_min*60*fs; % this window is different for scat coefs
win_step_samp = win_step_min*60*fs;
overlap_samp = win_samp - win_step_samp;

norder = 20;
fc = 170;
a = 1;
b = fir1(norder,fc/(fs/2), 'low');

data_all = filtfilt(b, a, double(data_all));


[aa_seg, aa_beg_eng ,~] = extractSegments(data_all, win_samp, overlap_samp);

% for each segment
for j = 1:size(aa_seg,2)
    ctemp_all{j, 1} = spm_ut_compute_feat_selected(aa_seg(:, j),'stat',fs);
end

Xtemp = spm_ut_flatten_segments(ctemp_all);

%%
figure
plot(Xtemp(:, 1))
a = axis;
grid on;
hold on;
