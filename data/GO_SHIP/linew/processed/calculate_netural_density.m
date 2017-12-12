%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to Calculate Netural Density on Line W using TEOS-10
% 
% Matlab Functions downloaded on October 17, 2017 from 
% http://www.teos-10.org/preteos10_software/neutral_density.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load GO-SHIP Line W Data 
data = csvread('/RESEARCH/chapter3/data/GO_SHIP/linew/section05.dat', 3);
LAT = data(:,2);
LON = data(:,3); 
CTDPRS = data(:,7);
CTDTMP = data(:,8);
CTDSAL = data(:,9);

gamma_n = eos80_legacy_gamma_n(CTDSAL,CTDTMP,CTDPRS,LON,LAT);

dlmwrite('/RESEARCH/chapter3/data/GO_SHIP/linew/processed/section05.gamma.csv',gamma_n,'delimiter',' ');