clc;
clear all;
close all; 
prt_number = 2;
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(prt_number)));

for d=1:1000
    Y=p{1,d};
    PPG_original=Y(1,300:800);
    BP_original=Y(2,300:800);
    
    BP = BP_original;
    PPG = PPG_original;
    
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence', max(PPG)/10,'MinPeakDistance',30); % max values of PPG signal
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence', max(PPG)/10,'MinPeakDistance',30); % min values of PPG signal
    dias_pk = -dias_pk;
    
    figure('Name',strcat('PPG and BP',int2str(d)))
    subplot(2,1,1);
    plot(PPG);
    hold on
    scatter(sys_loc, sys_pk)
    scatter(dias_loc, dias_pk)
    hold off

    subplot(2,1,2);
    plot(BP);
    hold on
    scatter(sys_loc, BP(sys_loc))
    hold off
    close all
end