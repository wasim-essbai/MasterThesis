clc;
clear all;
close all; 
prt_number = 4;
initial_indexes = [1 1 1 1166];
load(strcat('first_step_clean_record_indexes_',int2str(prt_number),'.mat'));
load(strcat('clean_record_indexes_',int2str(prt_number),'.mat'));
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(prt_number)));

Ts = 1/125;

initial_index = find(first_step_clean_record_indexes==initial_indexes(prt_number));
for d=initial_index:length(first_step_clean_record_indexes)
    Y=Part_4{1,first_step_clean_record_indexes(d)};
    PPG_original=Y(1,length(Y)/2-499:length(Y)/2+500);
    BP_original=Y(2, length(Y)/2-499:length(Y)/2+500);
    
    %% Signal filtering
    [b,a]=butter(4,[0.5*2*Ts,8*2*Ts]);
    PPG = filtfilt(b, a, PPG_original);
    PPG = PPG(200:800);
    
    [b,a]=butter(4,8*2*Ts);
    BP = filtfilt(b, a, BP_original);
    BP = BP(200:800);
    
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence', max(PPG)/8);
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence', max(PPG)/8);
    dias_pk = -dias_pk;
    
    [sys_bp_pk,sys_bp_loc]=findpeaks(BP, 'MinPeakProminence', max(BP)/8); 
    [dias_bp_pk,dias_bp_loc]=findpeaks(-BP, 'MinPeakProminence', max(BP)/8);
    dias_bp_pk = -dias_bp_pk;
    
    % Signal normalization
    max_ppg = max(PPG);
    min_ppg = min(PPG);
    PPG = (PPG - min_ppg)/(max_ppg - min_ppg);
    sys_pk = (sys_pk - min_ppg)/(max_ppg - min_ppg);
    dias_pk = (dias_pk - min_ppg)/(max_ppg - min_ppg);
    
    %% Signals plot
    figure_plot = figure('Name',strcat('PPG and BP',int2str(d)), 'visible','on');
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
    scatter(sys_bp_loc, sys_bp_pk)
    scatter(dias_bp_loc, dias_bp_pk)
    hold off
    
    f_input = figure('Name','Figure Input','visible','on');
    f_input.Position = [0 0 0 0];
    [~,~,button] = ginput(1);
    if(button == 32)
        clean_record_indexes = [clean_record_indexes first_step_clean_record_indexes(d)];
    end
    close all
end

% clean_record_indexes(clean_record_indexes==first_step_clean_record_indexes(d))=[];

clean_record_indexes = unique(clean_record_indexes);
save(strcat('clean_record_indexes_',int2str(prt_number),'.mat'),'clean_record_indexes','-append');