% Compute features on segments
% Supports only univariate features
%
% 1 = preictal
% 0 = interictal
clear all;
close all;
clc;

%% cluster
% nselect = 3;
% fprintf('Running feature computation\n');
% if nselect == 1; data_folders = {'train_1', 'test_1'}; end
% if nselect == 2; data_folders = {'train_2', 'test_2'}; end
% if nselect == 3; data_folders = {'train_3', 'test_3'}; end

addpath(genpath(pwd));

if strcmpi('jirka',getCurrentUser())
    path = '/home/jirka/data/kaggle_seisure_prediction_2016';
else
    if (MetaParPool('open') <= 0)
        disp('ERROR: Unable to initialize MetaParPool! Exiting...');
        exit(1);
    end
    path = '/storage/praha1/home/spilkjir/data/';
    addpath(genpath(path));
end

%% select data
data_folders = {'train_1', 'train_2', 'train_3', 'test_1', 'test_2', 'test_3'};

%% param
verbose = 2;
fs = 400;
N = 240000;
nr_ch = 16;

% the whole 10 min. segments
% win_min = 10; % window length for analysis

% sliding windows
win_min = 1; % window length for analysis
win_step_min = 0.5; % time step

win_samp = win_min*60*fs; % this window is different for scat coefs

win_step_samp = win_step_min*60*fs;
overlap_samp = win_samp - win_step_samp;

compute_univariate = 1;
compute_multivariate = 0;

%% select features - 10 minutes
sel_feat = 'stat'; nWin_limit = win_samp - 1;
% sel_feat = 'spectral'; nWin_limit = nWin_samp - 3072; % okno pro FFT je 2048
%sel_feat = 'mfj'; nWin_limit = nWin_samp - 3072; % okno pro FFT je 2048
%sel_feat = 'sp_entropy'; nWin_limit = nWin_samp - 3072; % okno pro FFT je 2048
% sel_feat = 'wav_entropy'; nWin_limit = nWin_samp - 3072; % okno pro FFT je 2048
%sel_feat = 'corr'; nWin_limit = nWin_samp - 1;

%% run

% feat - NaN prototype
if ~exist('featNaN','var')
    if compute_univariate
        featNaN = spm_ut_compute_feat_selected(nan(win_samp,1),sel_feat,fs);
    else
        featNaN = spm_ut_compute_feat_multivariate(nan(win_samp,16),sel_feat,fs);
    end
end

% get feature names
names = fieldnames(featNaN);
nr_feat = length(names);
aFeatNames = cell(1, nr_feat * nr_ch);
cnt = 0;
for i = 1:nr_ch
    for j = 1:nr_feat
        cnt = cnt + 1;
        aFeatNames{cnt} = sprintf('%d-%s',i,names{j});
    end
end

for idataf = 1:length(data_folders)
    
    sdate = datestr(now,30);
    sname = sprintf('sp2016_feat_%s_%s_%s.mat', data_folders{idataf}, sel_feat, sdate);
    sname = fullfile(path, sname);
    fprintf('Saving to: %s\n', sname);
    
    sdata = strcat(data_folders{idataf}, '_desc.mat');
    load(sdata);
    
    if verbose == 1
        fprintf('#### %s ####\n', data_folders{idataf});
    end
    
    %%%% % select data
    %ind = 1:size(data_desc,1);
    %aFiles = aFiles(ind);
    %data_desc = data_desc(ind,:);
    
    nr_channels = 16;
    y = data_desc(:,5);
    plabels = data_desc(:,7);
    data_quality = zeros(length(aFiles), 1);
    
    % number of segments
    r = win_samp - overlap_samp;
    kall = (N - overlap_samp)/r; % number of segments
      
    if compute_univariate
        c = spm_ut_compute_feat_selected(rand(N,1),sel_feat,fs);
        nr_feat = length(fieldnames(c));
        X_win = zeros(length(aFiles) * kall, nr_channels * nr_feat);
    end
    
    data_quality_win = zeros(length(aFiles) * kall, 1);
    y_win = zeros(length(aFiles) * kall, 1);
    plabels_win = zeros(length(aFiles) * kall, 1);
    plabels_10min = zeros(length(aFiles) * kall, 1);
    aFiles_win = cell(length(aFiles) * kall, 1);
    
    cnt_win = 0;
    cnt_10min_win = 0;
    for kk = 1:length(aFiles)
        
        if verbose > 1
            fprintf('%s, %d/%d, loading file: %s\n', data_folders{idataf}, ...
                kk, length(aFiles), aFiles{kk});
        end
        
        name = aFiles{kk};
        dd = load(name);
        data = dd.dataStruct.data;
        
        if isfield(dd.dataStruct, 'sequence')
            if dd.dataStruct.sequence ~= data_desc(kk,4)
                error('Records do not match');
            end
            
            yt = str2double(name(end-4:end-4));
            if yt ~= y(kk)
                error('Records do not match');
            end
        end
        
        %%%%%%% UNIVARIATE
        tic
        if compute_univariate
            
            q = zeros(kall, size(data,2));
            Xt = zeros(kall, nr_feat, size(data,2));
            % for each electrode
            parfor i = 1:size(data,2)
                
                % preprocessing
                x = data(:,i);
                x(x == 0) = NaN;
                
                [aa_seg, aa_beg_eng ,~] = extractSegments(x, win_samp, overlap_samp);
                ctemp_all = cell(size(aa_seg,2),1);
                
                % for each segment
                q_seg = zeros(size(aa_seg,2), 1);
                for j = 1:size(aa_seg,2)
                    
                    bnan = false;
                    seg = aa_seg(:, j);
                    
                    q_seg(j, 1) = 100*sum(~isnan(seg))/length(seg);
                    if sum(isnan(seg)) > nWin_limit
                        bnan = true;
                    end
                    
                    if bnan
                        ctemp_all{j} = featNaN;
                    else
                        ctemp_all{j} = spm_ut_compute_feat_selected(seg,sel_feat,fs);
                    end
                end
                
                Xtemp = spm_ut_flatten_segments(ctemp_all);
                Xt(:, :, i) = Xtemp;
                q(:, i) = q_seg(:, 1);
            end
                      
            % store data from individual windows
            cnt_10min_win = cnt_10min_win + 1;
            for j = 1:size(Xt, 1)
                
                cnt_win = cnt_win + 1;
                X_win(cnt_win,:) = Xt(j, :);
                
                data_quality_win(cnt_win) = mean(q(j,:));
                plabels_win(cnt_win) = plabels(kk);
                plabels_10min(cnt_win) = cnt_10min_win;
                y_win(cnt_win) = y(kk);
                aFiles_win{cnt_win} = aFiles{kk};
            end
        end
        toc
        
        if rem(kk, 500) == 0
            if exist('sname','var')
                s = sprintf('%s_%04d', sname, kk);
                save(s,'X_win', 'y_win', 'aFeatNames', 'aFiles_win', ...
                    'plabels_10min', 'plabels_win', 'data_quality_win', 'sel_feat', ...
                    'y', 'aFiles', 'data_desc', 'plabels', 'data_quality', ...
                    'win_min', 'win_step_min');
            end
        end
    end
    
    if exist('sname','var')
        save(sname,'X_win', 'y_win', 'aFeatNames', 'aFiles_win', ...
            'plabels_10min', 'plabels_win', 'data_quality_win', 'sel_feat', ...
            'y', 'aFiles', 'data_desc', 'plabels', 'data_quality', ...
            'win_min', 'win_step_min'); 
    end
    
end