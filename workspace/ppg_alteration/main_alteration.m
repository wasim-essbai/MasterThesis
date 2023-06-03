clear
clc
close all
prt_number = 2;
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(prt_number)));

alteration_type = 'gwn';
wgn_sigma = 0:0.1:2;

for i=1:length(wgn_sigma)
    full_feature_extraction_alteration(Part_2, alteration_type, wgn_sigma(i), i-1, prt_number);
end