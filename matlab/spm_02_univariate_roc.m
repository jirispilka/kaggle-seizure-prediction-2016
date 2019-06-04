% Compare AUC for inter-ictal vs preictal
% 
% 1 = preictal
% 0 = interictal
clear all;
close all;
clc;

sname = 'sp2016_feat_train_1_spectral_20160916.mat';
%sname = 'sp2016_feat_train_2_spectral_20160916.mat';
%sname = 'sp2016_feat_train_3_spectral_20160916.mat';

%sname = 'sp2016_feat_train_3_stat_20161031';

load(sname);
aFeatNames = aFeatNames';

% segmenty
% X = X_win;
% y = y_win;

for i =1:length(aFeatNames)
    aFeatNames{i} = strrep(aFeatNames{i},'_','-');
end

fprintf('\nData:              %d (%d, %d)\n', length(y),sum(y==0),sum(y==1));

% remove features
ind = (sum(isnan(X)) < 50); %& (sum(X == 0, 1) ~= size(X, 1));
%ind = (sum(isnan(X)) < 300); %& (sum(X == 0, 1) ~= size(X, 1));
X = X(:,ind);
fprintf('Features retained: %d, removed: %d\n\n', sum(ind), sum(~ind));
aFeatNames(~ind)
aFeatNames = aFeatNames(ind);

ind = sum(isnan(X),2) == 0;
% ind = sum(isnan(X),2 ) == 0 & sum(isinf(X),2 ) == 0;
X = X(ind, :);
y = y(ind);
%plabels = plabels(ind);
%data_quality = data_quality(ind);
fprintf('After nan removal: %d (%d, %d)\n\n', length(y), sum(y==0),sum(y==1));

%%
n = length(aFeatNames);
lw = 2;
fsize = 12;
color = distinguishable_colors(n);
cnt = 0;

figure
colAUC(X, y);
aAuc = colAUC(X, y);

for i = 1:size(X,2)
    x = double(X(:,i));
    mutual_info(i) = mi(x,y);
end

%%
figure(102)
stem(aAuc,'LineWidth',2)
grid on;

% set(gca,'XTick',1:length(aFeatNames))
% set(gca,'XTickLabel',aFeatNames)
% rotateXLabels( gca(), 90)
% a = axis;
% ylabel('AUC')
% axis([a(1) a(2) .5 1])
% set(gca,'FontSize',fsize)

figure
gscatter(aAuc, mutual_info);
grid on;

%%
selected_indices = feast('jmi',15,X,y);
selected_indices
aFeatNames(selected_indices)

createCorrMap(X(:,selected_indices),aFeatNames(selected_indices)');

figure
gscatter(aAuc(selected_indices), mutual_info(selected_indices));
grid on;


%%
figure
% id1 = 47;
% id2 = 48;
id1 = selected_indices(1);
id2 = selected_indices(2);
gscatter(X(:,id1), X(:,id2), y);
xlabel(aFeatNames(id1));
ylabel(aFeatNames(id2));
grid on;

