function [h,auc,p] = spm_ut_analyse_single_feat_stat(x, y, sTitle, aName, bPlot)
% Provides analysis of single feature 1) ROC curve, 2) boxplot, 3) p-value of hypothesis testings
%
% Inputs:
%  x  [nx1] feature vector
%  aIdxGroups [nx1] describes groups
%  sTitle [optional]
%  aGroupNames [optional]
%  pH [optional]
%
% Outputs:
%  h - handle to figure
%  AUC - area under ROC curve [float]
%  p - p-value of a statistical test [float]
%
% % Jiri Spilka, Patrice Abry, ENS Lyon 2014

if ~exist('sTitle','var')
    sTitle = '';
else
    sTitle = strrep(sTitle,'_','\_');
end

if ~exist('bPlot','var')
    bPlot = 1;
end

if ~exist('aName','var')
    t = unique(y);
    for i = 1:length(t)
        aName{i} = num2str(t(i));
    end
end

if length(unique(y)) > 2
    error('function do not support three classes');
end

% scrsz = get(0,'ScreenSize');
% h = figure('Position',[200 200 scrsz(3)/1.5 scrsz(4)*0.9]);

idx = ~isnan(x);
x = x(idx);
y = y(idx);

%% roc

if bPlot
    [xx,yy,~,auc] = perfcurve(double(y),x,1);
    
    if auc < 0.5
        t = (max(x)+1) - x;
        [xx,yy,~,auc] = perfcurve(double(y),t,1);
        %[xx,yy,~,auc] = perfcurve(double(aIdxGroups),aFeature,0);
        
        [~,idx] = unique(yy(:,1),'first');
        idx = [idx; length(yy)]; % include also last point
    end
else
    auc = colAUC(x, y);
end



%% p values
x1 = x(y);
y1 = x(~y);
p = ranksum(x1, y1);
stest = 'ranksum';

%% plotting

row = 1;
col = 2;

if bPlot
    scrsz = get(0,'ScreenSize');
    h = figure('Position',[200 200 scrsz(3)/2 scrsz(4)/2]);
    
    subplot(row,col,1)
    boxplot(x,y,'notch','on','Labels',aName)
    s = sprintf('%s \n test: %s, p=%1.2e',sTitle,stest,p);
    title(s)
    grid on;    
    
    subplot(row,col,2)
    plot(xx(idx,1),yy(idx,1),'b','LineWidth',2);
    grid on;
    s = sprintf('AUC=%1.2f',auc);
    set(gca,'XTick',0:0.1:1)
    set(gca,'YTick',0:0.1:1)
    title(s);
     
    
else
    h = 0;
end

