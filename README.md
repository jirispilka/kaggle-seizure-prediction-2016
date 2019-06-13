# Kaggle Seizure Prediction

Matlab and python codes for kaggle competitionL Seizure Prediction

I abandoned the competiont in the middle because of lack of trust 
in the data (train/test errors).  

The best results up to the 25.10.2016 were obtained spectrum features and 
spectral energy bands entropy with naive bayes classifier on top of that.

## Data
The dataset was only available for the competition.
https://www.kaggle.com/c/melbourne-university-seizure-prediction/data

## Matlab dependencies

Required toolbox:

- Carlos Granero-Belinchon, Stéphane G. Roux, Patrice Abry, Muriel Doret, Nicolas B. Garnier: Information Theory to Probe Intrapartum Fetal Heart Rate Dynamics. Entropy 19(12): 640 (2017)

- Stéphane Jaffard, Clothilde Melot, Roberto Leonarduzzi, Herwig Wendt, Patrice Abry Stéphane G. Roux, Maria E. Torres" p-exponent and p-leaders, Part I: Negative pointwise regularity, Physica A: Statistical Mechanics and its Applications
Volume 448, 15 April 2016, Pages 300-318

- MIToobox for C and MATLAB, http://www.cs.man.ac.uk/~pococka4/MIToolbox.html

- A Feature Selection Toolbox for C, Java and Matlab. - http://www.cs.man.ac.uk/~pococka4/FEAST.html

## Python

Classical machine learning stack: pandas, numpy, scipy, scikit-learn