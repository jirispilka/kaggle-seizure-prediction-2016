function [tot_pow,pow_split, pow_split2hz, alpha] = spm_features_spectral(X,fs,h,verbose)
% Routine for doing frequency analysis

%nfft = length(sig);
%f = (Fs/2)/nfft*(0:nfft-1); % frequency vector

% Instantiate spectrum object and call its PSD method.
if nargin < 3 || isempty(h)
    h = spectrum.welch;
    h.SegmentLength = 2^11;
end

if nargin < 4
    verbose = 0;
end

% subtract the mean
X = X - mean(X);

% JS: 2013-11-05 - compute PSD using welch method, the psd was not working (Matlab 2012a) 
win = h.SegmentLength;
overlap = round(h.SegmentLength*0.5);%h.SegmentLength/2;
[P,f] = pwelch(X,win,overlap,[],fs,'onesided');

% interpolate the spectrum to better find frequency bands
% just for the integration
% f_i = linspace(f(1),f(end), 2^5*length(f));
% P_i = interp1(f,P,f_i,'linear');
f_i = f;
P_i = P;

% % spectral slope
ind = (8 <= f_i & f_i <= 70);
p2 = polyfit(log2(f_i(ind)),log2(P_i(ind)),1);
bestFit = polyval(p2,log2(f_i));
alpha = -p2(:,1);

% trapeizodal interpolation
% the eps here is used to say lower than 0.003
pow_split(1) = computeInbandEnergy(f_i,P_i,0.1,4-eps); 
pow_split(2) = computeInbandEnergy(f_i,P_i,4,8-eps);
pow_split(3) = computeInbandEnergy(f_i,P_i,8,12-eps);
pow_split(4) = computeInbandEnergy(f_i,P_i,12,30-eps);
pow_split(5) = computeInbandEnergy(f_i,P_i,30,70-eps);
pow_split(6) = computeInbandEnergy(f_i,P_i,70,180);

tot_pow = sum(pow_split);

pow_split2hz(1) = computeInbandEnergy(f_i,P_i,0.1,2-eps); 
pow_split2hz(2) = computeInbandEnergy(f_i,P_i,2,4-eps);
pow_split2hz(3) = computeInbandEnergy(f_i,P_i,4,6-eps);
pow_split2hz(4) = computeInbandEnergy(f_i,P_i,6,8-eps);
pow_split2hz(5) = computeInbandEnergy(f_i,P_i,8,10-eps);
pow_split2hz(6) = computeInbandEnergy(f_i,P_i,10,12-eps);
pow_split2hz(7) = computeInbandEnergy(f_i,P_i,12,18-eps);
pow_split2hz(8) = computeInbandEnergy(f_i,P_i,18,24-eps);
pow_split2hz(9) = computeInbandEnergy(f_i,P_i,24,30-eps);
pow_split2hz(10) = computeInbandEnergy(f_i,P_i,30,40-eps);
pow_split2hz(11) = computeInbandEnergy(f_i,P_i,40,50-eps);
pow_split2hz(12) = computeInbandEnergy(f_i,P_i,50,60-eps);

ms = 12;
lw = 1;

if verbose > 0
    h = figure ; clf 
    hold on;
    plot(log2(f_i),log2(P_i), 'k','MarkerSize',ms,'LineWidth',lw); 
    a = axis;
    plot([log2(0.1) log2(0.1)],[a(3) a(4)],'--b','LineWidth',1)
    plot([log2(4) log2(4)],[a(3) a(4)],'--b','LineWidth',1)
    plot([log2(8) log2(8)],[a(3) a(4)],'--b','LineWidth',1)
    plot([log2(12) log2(12)],[a(3) a(4)],'--b','LineWidth',1)
    plot([log2(30) log2(30)],[a(3) a(4)],'--b','LineWidth',1)
    plot([log2(70) log2(70)],[a(3) a(4)],'--b','LineWidth',1)
    plot([log2(180) log2(180)],[a(3) a(4)],'--b','LineWidth',1)
    %plot(log2(f_i),bestFit, '--r','LineWidth',1)
    plot(log2(f_i(ind)),bestFit(ind), '-r','LineWidth',2)    
%     text(-12.5,-8,'ULF','color','b') ;
%     text(-8,-8,'VLF','color','b') ;
%     text(-4,-8,'LF','color','b') ;
%     text(-2.5,-8,'HF','color','b') ;
%     text(-1,-8,'VHF','color','b') ;

    xlabel('log(f)')
    ylabel('log(PSD)')
    legend('PSD')
%     s =  sprintf('Bands according to Task-Force\n');
%     s =  sprintf('%s ULF: %2.1f, VLF: %2.1f, LF: %2.1f, HF: %2.1f, HHF: %2.1f \n tot= %2.1f', ...
%         s,pow_split(1),pow_split(2),pow_split(3),pow_split(4),pow_split(5),pow_split(6));
    
%     title(s);
    grid on;  
    %fancyGraph(h);
end

%% compute band energy
function pow = computeInbandEnergy(f,Pxx,fl,fh)

ind = (fl <= f & f <= fh);
pow = trapz(f(ind),Pxx(ind));