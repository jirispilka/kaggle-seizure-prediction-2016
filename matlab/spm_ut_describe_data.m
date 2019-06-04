%% Kaggle 2016 Seisure prediction 2016
% Load the data from matfiles and create simple description
%
% Jiri Spilka, Prague 2016

clear all;
close all;
clc;

do_unzip = 0;
files = {'train_1', 'train_2', 'train_3', 'test_1', 'test_2', 'test_3'};

path = '/home/jirka/data/kaggle_seisure_prediction_2016';

for i = 1:length(files)
    
    file_name = files{i};
    s = fullfile(path, file_name);
    
    if do_unzip && ~exist(s, 'dir')
        system(sprintf('unzip %s', s));
        %unzip(s, path);
    end
           
    aFiles = getAllFiles(s);
    nrFiles = length(aFiles);   
    aFilesModified = cell(nrFiles, 1);
    
    data_desc = -1*ones(nrFiles, 7);
    
    for kk = 1:nrFiles
        fprintf('file_name: %s, file %d/%d\n', file_name, kk,nrFiles);
        name = aFiles{kk};
        load(name);
        
        tmp = getFileName(name);
        aFiles{kk} = tmp;
        tmp = split('_',tmp(1:end-4));
        
        if length(tmp) == 3
            %aFilesModified{kk} = sprintf('%d_%04d_%d',str2double(tmp{1}), ...
            %    str2double(tmp{2}),str2double(tmp{3}));
            aFilesModified{kk} = sprintf('%d_%d_%04d',str2double(tmp{1}), ...
                str2double(tmp{3}), str2double(tmp{2}));
        else
            aFilesModified{kk} = sprintf('%d_%04d',str2double(tmp{1}), ...
                str2double(tmp{2}));
        end
        
        data_desc(kk, 1) = dataStruct.nSamplesSegment;
        data_desc(kk, 2) = dataStruct.iEEGsamplingRate;
        data_desc(kk, 3) = length(dataStruct.channelIndices);
        
        if isfield(dataStruct, 'sequence')
            data_desc(kk, 4) = dataStruct.sequence;
            data_desc(kk, 5) = str2double(name(end-4:end-4));
        end
        
        % data quality
        d = dataStruct.data;
        q = zeros(1, size(d, 2));
        for j = 1:size(d, 2);
            q(j) = 100*sum(d(:,j)~=0)/length(d(:,j));
        end
        
        data_desc(kk, 6) = round2Decimal(mean(q),0);
        clear dataStruct
    end
           
    [aFilesModified, ind] = sort(aFilesModified);
    aFiles = aFiles(ind);
    data_desc = data_desc(ind,:);
    
    % add segment_id
    if data_desc(1,4) ~= -1
        segment_id = 1;
        for kk = 1:nrFiles
            data_desc(kk,7) = segment_id;
            if rem(kk,6) == 0
                if data_desc(kk,4) ~= 6
                    error('Something wrong here');
                end
                segment_id = segment_id + 1;
            end
        end
    end
    
    output_mat = strcat(file_name, '_desc.mat');
    save(output_mat, 'aFilesModified', 'aFiles', 'data_desc');
    
    output_file = strcat(file_name, '_desc.csv');
    fw = fopen(output_file, 'w+');
    
    fprintf(fw, 'name, name_modified, nr_samples, fs, channel_ids, sequence, class, avg. quality, segment id\n');
    for kk = 1:nrFiles
        fprintf(fw, '%s,', getFileName(aFiles{kk}));
        fprintf(fw, '%s,', aFilesModified{kk});
        for j = 1:size(data_desc, 2); 
            fprintf(fw, '%d', data_desc(kk, j));
            if j ~= size(data_desc, 2)
                fprintf(fw, ',');
            else
                fprintf(fw, '\n');
            end
        end
    end    
end

