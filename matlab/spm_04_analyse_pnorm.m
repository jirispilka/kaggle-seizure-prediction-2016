% Attempt to combine segments (windows) using p-norm
%
% 1 = preictal
% 0 = interictal
clear all;
close all;
clc;

sname = 'sp2016_feat_train_3_stat_20161101';

load(sname);
aFeatNames = aFeatNames';

% segmenty
X = X_win;

for i =1:length(aFeatNames)
    aFeatNames{i} = strrep(aFeatNames{i},'_','-');
end

fprintf('\nNumber of unique records: %d\n', length(y));
fprintf('\nNumber of windows: %d\n', size(X, 1));

% remove features
ind = (sum(isnan(X)) < 300); %& (sum(X == 0, 1) ~= size(X, 1));
X = X(:,ind);
aFeatNames = aFeatNames(ind);

% remove segments witn NaN
ind = sum(isnan(X),2) == 0;
X = X(ind, :);
y_win = y_win(ind);
plabels_win = plabels_win(ind);
fprintf('After nan removal: %d\n\n', length(y_win));

% remove corresponing files
% unames = unique(aFiles_win(~ind));
% files = zeros(length(unames),1);
% for i = 1:length(unames)
%     fun = @(x) strcmpi(x, unames(i));
%     tmp = find(cellfun(fun, aFiles));
%     if plabels10min
% end

% aAuc = colAUC(X, y);
ifeat = 1;
pp = [-Inf, -10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10, Inf];

up = unique(plabels_win);

for in = 1:length(pp)
    
    Xp = zeros(length(up), size(X, 2));
    yp = zeros(length(up), 1);

    
    for ifeat = 1:size(X, 2)
        for i = 1:length(up)
            ind = plabels_win == up(i);
            Xp(i,ifeat) = norm(X(ind, ifeat), pp(in));
            yp(i) = mean(y_win(ind));
            %if sum(y_win(ind) == 0 & y_win(ind) == 1) > 0
            %    error('Something wrong');
            %end          
        end
    end
    
    aAuc(in, :) = colAUC(Xp, yp);
    
end

%%
figure
plot(aAuc(:, 46), '--xr');
hold on;
plot(aAuc(:, 47), '--xk');
plot(aAuc(:, 48), '--xb');
grid on;
xlabel('p-norm')
ylabel('AUC')

%%
figure
plot(mean(aAuc, 2))
xlabel('p-norm')
ylabel('mean(AUC)')
grid on;
