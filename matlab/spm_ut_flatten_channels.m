function [X names_ch_feat] = spm_ut_flatten_channels(cfeat)
% return X of size 1 x [channels * features]

nr_ch = size(cfeat,1); % number of channels

% because some of the cell could be empty
for i = 1:length(cfeat)
    if ~isempty(cfeat{i})
        idxNotEmpty = i;
        break;
    end
end

names = fieldnames(cfeat{idxNotEmpty})';
nr_feat = length(names);

names_ch_feat = cell(1, nr_ch * nr_feat);
X = zeros(1, nr_ch * nr_feat);

for ich = 1:nr_ch
    cfeat_ch = cfeat{ich};
    
    if isempty(cfeat_ch)
        continue;
    end
    
    for k = 1:nr_feat
        ind = (ich-1)*nr_feat + k;
        names_ch_feat{ind} = sprintf('%d-%s',ich,names{k});
        X(1, ind) = cfeat_ch.(names{k});
    end
end