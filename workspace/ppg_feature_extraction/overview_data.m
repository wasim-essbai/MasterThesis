clc;
clear all;
close all; 
prt_number = 1;
initial_indexes = [106 1 390 1];
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(prt_number)));

Ts = 1/125;

for d=initial_indexes(prt_number):length(Part_1)
    Y=Part_1{1,d};
    PPG_original=Y(1,1:1000);
    BP_original=Y(2,1:1000);
    
    [b,a]=butter(4,[0.5*2*Ts,8*2*Ts]); % Bandpass digital filter design 
    PPG = filtfilt(b, a, PPG_original);
    %PPG = PPG_original;
    PPG = PPG(250:700);
    
    [b,a]=butter(4,8*2*Ts);
    BP = filtfilt(b, a, BP_original);
    %BP = BP_original;
    BP = BP(250:700);
    
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence', max(PPG)/10); % max values of PPG signal
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence', max(PPG)/10); % min values of PPG signal
    dias_pk = -dias_pk;
    
    [sys_bp_pk,sys_bp_loc]=findpeaks(BP, 'MinPeakProminence', max(BP)/10); 
    [dias_bp_pk,dias_bp_loc]=findpeaks(-BP, 'MinPeakProminence', max(BP)/10); % min value of BP(diastole) signal    
    dias_bp_pk = -dias_bp_pk;
    
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
    scatter(sys_bp_loc, sys_bp_pk)
    scatter(dias_bp_loc, dias_bp_pk)
    hold off
    close all


T =(0:Ts:(length(PPG)-1)*Ts); %time vector based on sampling rate

shift_index = 0;
    if(sys_loc(1,1) < dias_loc(1,1))
        shift_index=1;
    end 
output_record = [];
last_index = min([length(sys_loc)-1, length(dias_loc)-1]);
    for k=(1+shift_index):last_index
        sys_time = T(sys_loc(1,k)) - T(dias_loc(1,k-shift_index));
        dias_time = T(dias_loc(1,k+1-shift_index)) - T(sys_loc(1,k));
        cp = T(sys_loc(1,k+1)) - T(sys_loc(1,k));

        v = [0.1,0.25,0.33,0.5,0.66,0.75];

        ppg_st = zeros(1,length(v));
        ppg_dt = zeros(1,length(v));
        for j=1:length(v)
            for i=dias_loc(1,k-shift_index):sys_loc(1,k)
                if(PPG(1,i) >= (v(j)*(sys_pk(1,k) - dias_pk(1,k-shift_index)) + dias_pk(1,k-shift_index)))
                    lim_1=i;
                    break
                end
            end
   
            for i=sys_loc(1,k):dias_loc(1,k+1-shift_index)
                if(PPG(1,i) <= (v(j)*(sys_pk(1,k) - dias_pk(1,k+1-shift_index)) + dias_pk(1,k+1-shift_index)))
                    lim_2=i;
                    break
                end
            end

            ppg_st(j) = (sys_loc(1,k) - lim_1)*Ts;
            ppg_dt(j) = (lim_2 - sys_loc(1,k))*Ts;
        end
        filerow_features = [];
        for i=1:length(v)
            filerow_features = [filerow_features ppg_dt(i) ppg_st(i) ppg_st(i) + ppg_dt(i) ppg_dt(i)/ppg_st(i)];
        end
        
        filerow_features = [cp sys_time dias_time filerow_features];
        
        sys_bp_presence = 0;
        dias_bp_presence = 0;
        for j=1:length(sys_bp_loc)
            if(sys_bp_loc(j) >= sys_loc(1,k) && sys_bp_loc(j) <= sys_loc(1,k+1))
                sys_bp_presence = 1;
                break
            end
        end
        for j=1:length(dias_bp_loc)
            if(dias_bp_loc(j) >= sys_loc(1,k) && dias_bp_loc(j) <= sys_loc(1,k+1))
                dias_bp_presence = 1;
                break
            end
        end
        
        if((dias_bp_presence+dias_bp_presence) < 2)
            continue;
        end
        
        sbp = max(BP(1,sys_loc(1,k):sys_loc(1,k+1)));
        abp = min(BP(1,sys_loc(1,k):sys_loc(1,k+1)));
        filerow_target = [sbp abp];
        
        if(any(filerow_features <= 0) || any(isinf(filerow_features)) || sbp >= 180 || abp >= 130)
            output_record = [];
            samples_deleted = samples_deleted + 1;
            break;
        end
        
        output_record = [output_record; (d*prt_number*10000) filerow_target filerow_features];
    end
end