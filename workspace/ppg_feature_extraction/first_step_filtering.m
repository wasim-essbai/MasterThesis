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
for d=1:length(Part_4)
    Y=Part_4{1,d};
    PPG_original=Y(1,length(Y)/2-499:length(Y)/2+500);
    BP_original=Y(2, length(Y)/2-499:length(Y)/2+500);
    
    %% Signal Filtering
    [b,a]=butter(4,[0.5*2*Ts,8*2*Ts]);
    PPG = filtfilt(b, a, PPG_original);
    PPG = PPG(200:800);
    
    T =(0:Ts:(length(PPG)-1)*Ts); %time vector based on sampling rate
    
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence',max(PPG)/8);
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence',max(PPG)/8);
    dias_pk = -dias_pk;
    
    [b,a]=butter(4,8*2*Ts);
    BP = filtfilt(b, a, BP_original);
    BP = BP(200:800);

    [sys_bp_pk,sys_bp_loc]=findpeaks(BP, 'MinPeakProminence', max(BP)/8); 
    [dias_bp_pk,dias_bp_loc]=findpeaks(-BP, 'MinPeakProminence', max(BP)/8);
    dias_bp_pk = -dias_bp_pk;
   
    shift_index = 0;
    if(sys_loc(1,1) < dias_loc(1,1))
        shift_index=1;
    end
    
    %% Check if peaks are detected correctly
    
    if(abs(length(dias_loc) - length(sys_loc)) >  4)
        good = 0;
        bad_peaks = bad_peaks + 1;
        samples_deleted = samples_deleted + 1;
        continue;
    end
    
    if(any(sys_pk < mean(sys_pk)/7))
        output_record = [];
        samples_deleted = samples_deleted + 1;
        continue;
    end
    
    if(any(abs(dias_pk) > abs(mean(dias_pk)*3)))
        output_record = [];
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

%save(strcat('first_step_clean_record_indexes_',int2str(prt_number),'.mat'),'first_step_clean_record_indexes','-append');
save(strcat('first_step_clean_record_indexes_',int2str(prt_number),'.mat'),'first_step_clean_record_indexes');
close(f);
toc
disp('samples_deleted')
disp(samples_deleted)
disp('bad_peaks')
disp(bad_peaks)