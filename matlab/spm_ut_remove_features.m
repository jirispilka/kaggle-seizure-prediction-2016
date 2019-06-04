% remove features with many NaNs
% remove non-robust features

clear all;
close all;
clc;

% s = 'sp2016_feat_%s_%d_mfj_20160926';
% cremove = {'-c3_j_', '-c4_j_'};
cretain = {'_H_sh_knn'};

cremove = {};
% s = 'sp2016_feat_%s_%d_stat_20160915.mat';
% s = 'sp2016_feat_%s_%d_spectral_20160916.mat';
s = 'sp2016_feat_%s_%d_sp_entropy_20160919.mat';
% s = 'sp2016_feat_%s_%d_corr_20161001.mat';
% s = 'sp2016_feat_%s_%d_wav_entropy_20161019.mat';

bsave = 1;

if ~isempty(cretain) && ~isempty(cremove)
    error('Only remove or retain allowed')
end

for nsubject = 2:3
    
    sname = sprintf(s, 'train', nsubject);
    sname_test = sprintf(s, 'test', nsubject);
    
    fprintf('############## Loading file %s\n', sname)
    fprintf('############## Loading file %s\n', sname_test)
    
    load(sname);
    data_test = load(sname_test);
    
    if ~isempty(cretain)
        ind_remove = ones(1, length(aFeatNames));
    else
        ind_remove = zeros(1, length(aFeatNames));
    end
    
    fprintf('\nData: %d, %d\n', size(X,1), size(X,2));
    
    ind = (sum(isnan(X)) > 50);
    ind_remove(ind == 1) = 1;
    
    sum(isnan(data_test.X));
    ind = (sum(isnan(data_test.X)) > 60);
    ind_remove(ind == 1) = 1;
    
    aFeatNames(ind);
    
    if ~isempty(cremove)
        for k = 1:length(cremove)
            name = cremove{k};
            fun = @(x) ~isempty(strfind(x,name));
            ind = find(cellfun(fun,aFeatNames));
            ind_remove(ind) = 1;
        end
    elseif ~isempty(cretain)
        for k = 1:length(cretain)
            name = cretain{k};
            fun = @(x) ~isempty(strfind(x,name));
            ind = find(cellfun(fun,aFeatNames));
            ind_remove(ind) = 0;
        end
    end
    
    ind_remove = ind_remove == 1;
    %aFeatNames(ind_remove)
    
    X = X(:, ~ind_remove);
    aFeatNames = aFeatNames(~ind_remove);
    
    if bsave
        save(sname,'X','y', 'aFeatNames' , 'aFiles', 'data_desc', ...
            'sel_feat', 'plabels', 'data_quality');
    end
    
    load(sname_test);
    X = X(:, ~ind_remove);
    aFeatNames = aFeatNames(~ind_remove);
    
    if bsave
        save(sname_test,'X','y', 'aFeatNames' , 'aFiles', 'data_desc', ...
            'sel_feat', 'plabels', 'data_quality');
    end
    
    fprintf('Features removed: %d\n\n', sum(ind_remove));
    
end

