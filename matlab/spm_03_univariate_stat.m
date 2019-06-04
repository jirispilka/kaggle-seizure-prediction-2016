% Analyse univariate feature statistics (AUC and p-value)
%
% Jiri Spilka, Prague 2016

%clear all;
close all; clc;

bsingle_file = 0;

%% single file

% if bsingle_file
%     npatient = 1;
%     sname = sprintf('sp2016_feat_train_%d_soe_20160912', npatient);
%     fprintf('name: %s\n', sname);
%     
%     load(sname);
%     aFeatNames = aFeatNames';
%     nNrFeat = length(aFeatNames);
%     
%     for ifeat = 1:3%nNrFeat
%         x = X(:, ifeat);
%         spm_ut_analyse_single_feat_stat(x, y==1, aFeatNames{ifeat})
%     end
%     
%     return
% end

%%

for ip = 1:3
    
    sname = sprintf('sp2016_feat_train_%d_sp_entropy_20160919', ip);
    
    fprintf('name: %s\n', sname);
    
    load(sname);
    aFeatNames = aFeatNames';
    nNrFeat = length(aFeatNames);
    
    fprintf('recs: y=1: %d, y=0: %d \n', sum(y == 1), sum(y == 0));
    
    X_all_p{ip} = X;
    y_all_p{ip} = y;
end

spm_ut_analyse_3p_print_tex(X_all_p, aFeatNames, 'test.tex', y_all_p)


%%
