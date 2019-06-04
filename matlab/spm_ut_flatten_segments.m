function [X names_feat] = spm_ut_flatten_segments(cfeat)

nr_seg = size(cfeat,1); % number of channels

% because some of the cell could be empty
for i = 1:length(cfeat)
    if ~isempty(cfeat{i})
        idxNotEmpty = i;
        break;
    end
end

names_feat = fieldnames(cfeat{idxNotEmpty})';
nr_feat = length(names_feat);

X = zeros(nr_seg, nr_feat);

for iseg = 1:nr_seg
    cfeat_ch = cfeat{iseg};
    
    if isempty(cfeat_ch)
        continue;
    end
    
    for k = 1:nr_feat
        X(iseg, k) = cfeat_ch.(names_feat{k});
    end
end