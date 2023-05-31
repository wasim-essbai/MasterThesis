clc;
clear all;
close all; 
prt_number = 4;
bad_peaks=0;
samples_deleted=0;
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(prt_number)));

Ts=1/125;

first_step_clean_record_indexes = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
f = waitbar(0,'Cleaning at first stage...');
for d=1:3000
    Y=Part_4{1,d};
    PPG_original=Y(1,:);
    BP_original=Y(2,:);
    
    if(size(PPG_original,2) < 25500)
        continue
    end
    
    %% Signal Filtering
    [b,a]=butter(4,[0.5*2*Ts,8*2*Ts]);
    PPG = filtfilt(b, a, PPG_original);
    
    max_ppg = max(PPG);
    min_ppg = min(PPG);
    PPG = (PPG - min_ppg)/(max_ppg - min_ppg);
    
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence',max(PPG)/6,'MinPeakDistance',22);
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence',max(PPG)/6,'MinPeakDistance',25);
    dias_pk = -dias_pk;
    
    [b,a]=butter(4,8*2*Ts);
    BP = filtfilt(b, a, BP_original);
    [BP,i,bpmedian,bpsigma] = hampel(BP);
    
    [sys_bp_pk,sys_bp_loc]=findpeaks(BP, 'MinPeakProminence', max(BP)/8); 
    [dias_bp_pk,dias_bp_loc]=findpeaks(-BP, 'MinPeakProminence', max(BP)/8);
    dias_bp_pk = -dias_bp_pk;
   
    shift_index = 0;
    if(sys_loc(1,1) < dias_loc(1,1))
        shift_index=1;
    end
    
%     plot(PPG)
%     hold on
%     scatter(sys_loc,sys_pk)
%     scatter(dias_loc,dias_pk)
%     hold off
%     close all
    %% Check if peaks are detected correctly
    
    if(abs(length(dias_loc) - length(sys_loc)) >  4)
        good = 0;
        bad_peaks = bad_peaks + 1;
        samples_deleted = samples_deleted + 1;
        continue;
    end
    
    if(any(sys_pk < mean(sys_pk)/4))
        output_record = [];
        samples_deleted = samples_deleted + 1;
        continue;
    end
    
    if(any(dias_pk > mean(dias_pk)*4))
        output_record = [];
        samples_deleted = samples_deleted + 1;
        continue;
    end
    
    if(abs(length(dias_bp_loc) - length(sys_bp_loc)) >  4)
        good = 0;
        bad_peaks = bad_peaks + 1;
        samples_deleted = samples_deleted + 1;
        continue;
    end
    
    if(isempty(sys_bp_loc) || isempty(dias_bp_loc))
        output_record = [];
        samples_deleted = samples_deleted + 1;
        continue;
    end
    
    good = 1;
    sys = 0;
    dias_counter = 1;
    sys_counter = 1 + shift_index;
    for j=1:(length(sys_loc) + length(dias_loc) - shift_index - 2)
        if(sys)
            if(sys_loc(1,sys_counter) > dias_loc(1,dias_counter))
                good = 0;
                break;
            end
            sys_counter = sys_counter + 1;
            sys=0;
        else
            if(sys_loc(1,sys_counter) < dias_loc(1,dias_counter))
                good = 0;
                break;
            end
            dias_counter = dias_counter + 1;
            sys=1;
        end
    end
    
    if(good == 0)
        bad_peaks = bad_peaks + 1;
        samples_deleted = samples_deleted + 1;
        continue
    end
    
    first_step_clean_record_indexes=[first_step_clean_record_indexes d];
    waitbar((d-1)/3000,f,'Cleaning at first stage...');
end

save(strcat('full_first_step_clean_record_indexes_',int2str(prt_number),'.mat'),'first_step_clean_record_indexes','-append');
%save(strcat('full_first_step_clean_record_indexes_',int2str(prt_number),'.mat'),'first_step_clean_record_indexes');
close(f);
toc
disp('samples_deleted')
disp(samples_deleted)
disp('bad_peaks')
disp(bad_peaks)