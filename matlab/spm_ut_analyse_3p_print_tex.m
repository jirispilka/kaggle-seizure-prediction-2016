function spm_ut_analyse_3p_print_tex(cX, aFeatNames, sTexName, cy)
% Creates TeX file of features with table of AUC and p-values.
%
% Once the file is generated, it should be included into Tex document.
% A template of the document is present at the end of this file.
% Then use normal latex commands.
%
% Inputs:
%
% Outputs:
%  out is file: sTexName
%
% % Jiri Spilka, Patrice Abry, ENS Lyon 2014

nNrFeat = length(aFeatNames);
fid = fopen(sTexName,'w+');

%% print table
fprintf(fid,'%s',printHeader());

for i = 1:nNrFeat
    name = aFeatNames{i};
    
    name = strrep(name,'_','\_');
    fprintf(fid,'%20s ',name);
    
    for k = 1:length(cX)
        Xt = cX{k};
        y = cy{k} == 1;
        
        x = Xt(:,i);
        [~, auc, pvals] = spm_ut_analyse_single_feat_stat(x, y == 1, '', name, 0);
        if pvals < 0.05
            s = '\textsuperscript{*}';
        else
            s = '';
        end
        
        fprintf(fid,' & \\boxProb{%3d}{%s}', round(100*auc),strip(abs(auc),'%1.2f'));
        fprintf(fid,' & %2.2e%20s', pvals,s);
    end

    fprintf(fid,'\\\\\n');
end

fprintf(fid,'%s',printFooter);
fprintf(fid,'\n');

fprintf('DONE\n')

%% functions
function s = printFooter()
s = sprintf('\\bottomrule\n');
s = sprintf('%s\\end{tabular}\n',s);
%s = sprintf('%s\\caption{\\small{\\textsuperscript{*} p $<$ 0.05}}\n',s);
s = sprintf('%s\\end{table}\n',s);

function s = printHeader()
s = sprintf('\\begin{table}[ht!]\\small\n');
s = sprintf('%s\\centering\n',s);
s = sprintf('%s\\begin{tabular}{ l  rl l rl l rl l}\n',s);
s = sprintf('%s\\toprule \n',s);
s = sprintf('%s  & \\multicolumn{3}{c}{patient 1} & \\multicolumn{3}{c}{patient 2} & \\multicolumn{3}{c}{patient 3}\\\\\n',s);
s = sprintf('%s \\cmidrule(r){2-4} \n',s);
s = sprintf('%s \\cmidrule(l){5-7} \n',s);
s = sprintf('%s \\cmidrule(r){8-10} \n',s);
colnames = ' \multicolumn{2}{c}{AUC} & $p$ & \multicolumn{2}{c}{AUC} & $p$ & \multicolumn{2}{c}{AUC} & $p$';
s = sprintf('%s name & %s \\\\\n',s,colnames);
s = sprintf('%s\\midrule\n',s);

%% images
% function s = printFigure(name, size)
% s = sprintf('\\begin{figure}[ht!]\n');
% s = sprintf('%s\\centering\n',s);
% s = sprintf('%s\\includegraphics[width=%1.1f\\textwidth]{images/%s}\n',s, size, name);
% s = sprintf('%s\\caption{%s}\n',s,name);
% s = sprintf('%s\\label{fig:%s}\n',s,name);
% s = sprintf('%s\\end{figure}\n\n',s);

function s = strip(val, format)

s = sprintf(format,val);
s = s(2:end);

%% TEX template
% \documentclass[a4paper,11pt,twoside]{book}
% \usepackage[english]{babel}
% \usepackage[utf8]{inputenc}
% \usepackage{graphicx,color}
% \usepackage[top=3cm, bottom=3cm, right=2.5cm, left=2.5cm]{geometry}% set boundary
% \usepackage[T1]{fontenc}
% \usepackage{times}
%
% \usepackage[colorlinks=true,pagebackref=true]{hyperref}
% \hypersetup{
%    citecolor={black},
%    linkcolor={blue},
%    urlcolor={black},
% }
%
% \usepackage{multirow} % multirow v tabulce
% \usepackage{booktabs}
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% TIKZ
% \usepackage{tikz}
% % for a table -- also shows graphical representation
% \newcommand{\boxProb}[2]
% {
% \begin{tikzpicture}
% \def\w{2}
% \def\x{#1/100*\w}
% \filldraw[fill=gray!#1!white, draw=black] (0,0) rectangle (\x,0.2);
% \draw [gray] (0,0) rectangle (\w,0.2);
% \end{tikzpicture} & #2
% }
%
% \begin{document}
% \pagenumbering{arabic}
% \pagestyle{plain}
%
% \hfill {Generated on: \today}
% \vglue 1cm
%
% \textbf{Description: Analysis LF/HF on large database, currently: 2079 records.}
%
% \begin{itemize}
%   \item pathological $pH \leq 7.05$
%   \item known issue: when arterial pH was missing the venouse was used instead. This might cause some false positives.
% \end{itemize}
%
% Frequency spectrum estimation:
% \begin{itemize}
%   \item welch method
%   \item window length = 1024 samples, overlap 80\%
% \end{itemize}
%
% \vglue 1cm
%
% \textbf{LF/HF bands and scaling for H: }
%
% \begin{verbatim}
% LFband1 = [0.04 0.15] ; % J = [6, 7]
% HFband1 = [0.15 0.40] ; % J = [4, 5]
%
% LFband2 = [0.04 0.15] ; % J = [6, 7]
% HFband2 = [0.15 1.00] ; % J = [3, 5]
%
% LFband3 = [0.04 0.23] ;
% HFband3 = [0.23 0.40] ;
%
% LFband4 = [0.04 0.31] ;
% HFband4 = [0.31 0.40] ;
%
% LFband5 = [0.015 0.125] ;  %  J = [6, 7, 8]
% HFband5 = [0.125 1.000] ;  %  J = [3, 4, 5]
%
% LFband6 = [0.015 0.25] ; %  J = [5, 6, 7, 8]
% HFband6 = [0.250 1.00] ; %  J = [3, 4]
%
% \end{verbatim}
%
% \include{analysis_130111_largeDb_2079}
%
% \end{document}