function [coef, leaders, nj] = DxLx1d_js(data, Nwt, gamint)
% Calculate DWT and leaders (small modifs by JS)
% HW Lyon 05/2007

Norm=1;

if nargin<3; gamint=0; end;

%-- Initialize the wavelet filters
n = length(data) ;                   % data length
hh1 = fliplr(rlistcoefdaub(Nwt)) ;      % scaling function filter
nl = length(hh1) ;                    % length of filter, store to manage edge effect later
gg1 = fliplr((-1).^(0:-1+nl).*hh1) ;  % wavelet filter
%--- Predict the max # of octaves available given Nwt, and take the min with
nbvoies= fix( log2(length(data)) );
nbvoies = min( fix(log2(n/(2*Nwt+1)))  , nbvoies);  %   safer, casadestime having problems
%--- Compute the WT, calculate statistics
njtemp = n;    %  this is the the appro, changes at each scale
approW=data;
approL=data;
for j=1:nbvoies         % Loop Scales
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% PHASE 1: COEFFICIENTS, LEADERS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %-- Phase 1a: get the wavelet coefficients at this scale
    njtemp = length(approW) ;
    conv_gg1 = conv(approW,gg1) ;
    conv_hh1 = conv(approW,hh1) ;
    approW = conv_hh1(nl:2:njtemp) ; %-- prepare for Phase 1 at next coarser scale
    decime = conv_gg1(nl:2:njtemp) ;   % details at scale j. decime always becomes empty near the end,
    nj.W(j) = length(decime);            % number of coefficients
    if nj.W(j) == 0                      % forget this j and exit if no coefficients left!
        fprintf('In wtspecq_statlog, oops!  no details left at scale %d! \n',j)
        %break
    end
    AbsdqkW = abs(decime)*2^(j/2)/2^(j/Norm);   %%%%%%% passage Norme L1
    %% ----> ADDED 16/04/2007
    % hmin before integration
    coef(j).supcoefnoint=max(AbsdqkW);
    % fractional integration
    AbsdqkW = AbsdqkW*2^(gamint*j);
    [coef(j).supcoef, coef(j).supcoefid]=max(AbsdqkW);
    coef(j).approW=(approW)*2^(j/2)/2^(j/Norm)*2^(gamint*j);
    coef(j).value=AbsdqkW;%(find(AbsdqkW>=eps));
    coef(j).value_noabs=(decime)*2^(j/2)/2^(j/Norm)*2^(gamint*j); % MODIF PA 2010 MAY 7
    coef(j).gamma=gamint;
    coef(j).sign=sign(decime);
    %% <---- ADDED 16/04/2007
    
    %-- Phase 1b: calculate the wavelet leaders at this scale
    if j==1
        fin_effet_bord =  length(approL) ;
        approL = conv_hh1(nl:2:fin_effet_bord) ;
        decime = abs(conv_gg1(nl:2:fin_effet_bord) ) *2^(j/2)/2^(j/Norm); %%%%%%% passage Norme L1
        %% ----> ADDED 16/04/2007
        % fractional integration
        decime = decime*2^(gamint*j);
        %% <---- ADDED 16/04/2007
        leaders(1).sans_voisin.value = decime;
        leaders(1).value = max([decime(1:end-2);decime(2:end-1);decime(3:end)]);
        fin_effet_bord =  length(approL) ;
        AbsdqkL=leaders(1).value;
        nj.L(j) = length(AbsdqkL);
    else
        approL = conv_hh1(nl:2:fin_effet_bord) ;
        decime = abs(conv_gg1(nl:2:fin_effet_bord) ) *2^(j/2)/2^(j/Norm) ; %%%%%%% passage Norme L1
        %% ----> ADDED 16/04/2007
        % fractional integration
        decime = decime*2^(gamint*j);
        %% <---- ADDED 16/04/2007
        length_detail = 2*length(decime);
        %leaders(j).sans_voisin.value = max([decime; leaders(j-1).sans_voisin.value(1:2:length_detail); leaders(j-1).sans_voisin.value(2:2:length_detail)]);
        leaders(j).sans_voisin.value = max([decime; leaders(j-1).sans_voisin.value(nl-1:2:length_detail+nl-2); leaders(j-1).sans_voisin.value(nl:2:length_detail+nl-2)]);
        leaders(j).value = max([leaders(j).sans_voisin.value(1:end-2);leaders(j).sans_voisin.value(2:end-1);leaders(j).sans_voisin.value(3:end)]);
        AbsdqkL=leaders(j).value;
        nj.L(j) = length(AbsdqkL);
        if nj.L(j) == 0                      % forget this j and exit if no coefficients left!
            fprintf('In wtspecq_statlog, oops!  no details left at scale %d! \n',j)
            %break
        end
        fin_effet_bord = length(approL) ;
    end
    leaders(j).gamma=gamint;    
    [leaders(j).mincoef, leaders(j).mincoefid]=min(leaders(j).value);
    [leaders(j).supcoefL, leaders(j).supcoefidL]=max(leaders(j).value);
    leaders(j).supcoefnoint=coef(j).supcoefnoint; leaders(j).supcoef=coef(j).supcoef; leaders(j).supcoefid=coef(j).supcoefid;
    coef(j).mincoef=leaders(j).mincoef; coef(j).mincoefid=leaders(j).mincoefid;
    coef(j).supcoefL=leaders(j).supcoefL; coef(j).supcoefidL=leaders(j).supcoefidL;
end