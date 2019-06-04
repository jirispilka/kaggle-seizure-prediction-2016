function spm_ut_eegplot(data, channels)

if ~exist('channels', 'var')
    channels = 1:size(data, 2);
end

color = distinguishable_colors(16);

figure(1435)
hold on;
for i = 1:size(data, 2)
    if any(channels == i)
        plot(data(:,i) + i*1000, 'color', color(i,:))
        %plot(data(:,i) + i*1000, 'color', 'k')
        grid on
        axis tight
    end
 
end