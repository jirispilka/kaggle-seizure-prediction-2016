% Compute different features and save them into mat files
%
% 1 = preictal
% 0 = interictal
clear all;
close all;
clc;

%%
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

% the whole 10 min. segments
win_min = 10; % window length for analysis
nWin_samp = win_min*60*fs; % this window is different for scat coefs

compute_univariate = 1;
compute_multivariate = 0;

%% select features - 10 minutes
sel_feat = 'stat'; nWin_limit = nWin_samp - 1;
% sel_feat = 'spectral'; nWin_limit = nWin_samp - 3072; % okno pro FFT je 2048
%sel_feat = 'mfj'; nWin_limit = nWin_samp - 3072; % okno pro FFT je 2048
%sel_feat = 'sp_entropy'; nWin_limit = nWin_samp - 3072; % okno pro FFT je 2048
% sel_feat = 'wav_entropy'; nWin_limit = nWin_samp - 3072; % okno pro FFT je 2048
%sel_feat = 'corr'; nWin_limit = nWin_samp - 1;

%% run

% feat - NaN prototype
if ~exist('featNaN','var')
    if compute_univariate
        featNaN = spm_ut_compute_feat_selected(nan(nWin_samp,1),sel_feat,fs);
    else
        featNaN = spm_ut_compute_feat_multivariate(nan(nWin_samp,16),sel_feat,fs);
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
        
    y = data_desc(:,5);
    plabels = data_desc(:,7);
    data_quality = zeros(length(aFiles),1);
    
    nr_channels = 16;
    
    %%%%% allocate data matrix
    if compute_multivariate
        c = spm_ut_compute_feat_multivariate(rand(240000,16),sel_feat,fs);
        nr_feat = length(fieldnames(c));
        X = zeros(length(aFiles), nr_feat);
    end
    
    if compute_univariate
        c = spm_ut_compute_feat_selected(rand(240000,1),sel_feat,fs);
        nr_feat = length(fieldnames(c));
        X = zeros(length(aFiles), nr_channels * nr_feat);
    end
    
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
        
        %%%%%%%% MULTIVARIATE
        if compute_multivariate == 1
            
            % just compute data quality
            % replace zero by NaNs
            q = zeros(size(data,2),1);
            for i = 1:size(data,2)
                data(data(:,i) == 0,i) = NaN;
                q(i) = 100*sum(~isnan(data(:,i)))/length(data(:,i));
                if sum(isnan(data(:,i))) > nWin_limit
                    bnan = true;
                end
            end
            data_quality(kk) = mean(q);
            
            cfeat = spm_ut_compute_feat_multivariate(data, sel_feat, fs);
            [Xt aFeatNames] = spm_ut_flatten_channels({cfeat});
            X(kk,:) = Xt;
            

        end
        
        %%%%%%% UNIVARIATE
        tic
        if compute_univariate
            ctemp_all = cell(size(data,2),1);
            q = zeros(size(data,2),1);
            parfor i = 1:size(data,2)
                
                %fprintf('processing %d/%d\n', i, size(data,2));
                x = data(:,i);
                
                % preprocessing
                x(x == 0) = NaN;
                bnan = false;
                
                q(i) = 100*sum(~isnan(x))/length(x);
                %disp(q)
                if sum(isnan(x)) > nWin_limit
                    bnan = true;
                end
                
                % compute on complete data
                if bnan
                    ctemp_all{i} = featNaN;
                else
                    ctemp_all{i} = spm_ut_compute_feat_selected(x,sel_feat,fs);
                end
                % compute on windows
                %ctemp_all{i} = spm_ut_compute_feat_selected(x,sel_feat,fs);
            end
            [Xt aFeatNames] = spm_ut_flatten_channels(ctemp_all);
            X(kk,:) = Xt;
            data_quality(kk) = mean(q);
        end
        toc
        
        if rem(kk, 500) == 0
            if exist('sname','var')
                s = sprintf('%s_%04d', sname, kk);
                save(s,'X','y', 'aFeatNames' , 'aFiles', 'data_desc', ...
                    'sel_feat', 'plabels', 'data_quality');
            end
        end
    end
    
    if exist('sname','var')
        save(sname,'X','y', 'aFeatNames' , 'aFiles', 'data_desc', ...
            'sel_feat', 'plabels', 'data_quality');
    end
    
end