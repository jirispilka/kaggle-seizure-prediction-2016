% Plot progression of some feature
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
% ind = sum(isnan(X),2) == 0;
% % ind = sum(isnan(X),2 ) == 0 & sum(isinf(X),2 ) == 0;
% X = X(ind, :);
% y_win = y_win(ind);
% plabels_win = plabels_win(ind);
% fprintf('After nan removal: %d\n\n', length(y_win));


ifeat = 48; % SELECT FEATURE TO BE USED
aFeatNames{ifeat}

x = X_win(:, ifeat);
N = length(x);

%%

figure(11); clf

subplot(211)
hold on;
plot(x(y_win == 0), 'b')
grid on;
xlabel('time')

subplot(212)
plot(x(y_win == 1), 'r')
a = axis;
grid on;
xlabel('time')

%%
% uname = aFiles_win{11060};
% fun = @(x) strcmpi(x, uname);
% tmp = find(cellfun(fun, aFiles));
% tmp
