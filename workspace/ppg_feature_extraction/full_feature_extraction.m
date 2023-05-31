clc;
clear all;
close all; 
prt_number = 2;
load(strcat('full_clean_record_indexes',int2str(prt_number)));
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(prt_number)));

output_file=[];
samples_deleted = 0;
bad_peaks = 0;
added=[];

filerow_header = ["ID" "mbp" "sbp" "dbp"];
filerow_header = [filerow_header "cp" "sut" "dt"];
filerow_header = [filerow_header "dt10" "st10" "st10_p_dt10" "st10_d_dt10"];
filerow_header = [filerow_header "dt25" "st25" "st25_p_dt25" "st25_d_dt25"];
filerow_header = [filerow_header "dt33" "st33" "st33_p_dt33" "st33_d_dt33"];
filerow_header = [filerow_header "dt50" "st50" "st50_p_dt50" "st50_d_dt50"];
filerow_header = [filerow_header "dt66" "st66" "st66_p_dt66" "st66_d_dt66"];
filerow_header = [filerow_header "dt75" "st75" "st75_p_dt75" "st75_d_dt75"];
output_file = [output_file;filerow_header];


Ts=1/125;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
f = waitbar(0,'Extracting features...');
for d=1:length(full_clean_record_indexes)
    Y=Part_2{1,full_clean_record_indexes(d)};
    signal_limit = min(size(Y(1,:),2),52500);
    PPG_original=Y(1,1:signal_limit);
    BP_original=Y(2,1:signal_limit);
  
    [b,a]=butter(4,[0.5*2*Ts,8*2*Ts]);
    PPG = filtfilt(b, a, PPG_original);
    
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence',max(PPG)/6,'MinPeakDistance',22);
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence',max(PPG)/6,'MinPeakDistance',25);
    dias_pk = -dias_pk;
    
    % Signal normalization
    max_ppg = max(PPG);
    min_ppg = min(PPG);
    PPG = (PPG - min_ppg)/(max_ppg - min_ppg);
    sys_pk = (sys_pk - min_ppg)/(max_ppg - min_ppg);
    dias_pk = (dias_pk - min_ppg)/(max_ppg - min_ppg);
    
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
    
    output_record = [];
    last_index = min([length(sys_loc)-1, length(dias_loc)-1]);
    do_average = 0;
    for k=(1+shift_index):last_index
        sys_time = (sys_loc(1,k) - dias_loc(1,k-shift_index))*Ts;
        dias_time = (dias_loc(1,k+1-shift_index) - sys_loc(1,k))*Ts;
        cp = (sys_loc(1,k+1) - sys_loc(1,k))*Ts;

        v = [0.1,0.25,0.33,0.5,0.66,0.75];
        
        ppg_st = zeros(1,length(v));
        ppg_dt = zeros(1,length(v));
        for j=1:length(v)
            lim_1 = 0;
            lim_2 = 0;
            for i=dias_loc(1,k-shift_index):sys_loc(1,k)
                if(PPG(1,i) >= ((v(j) - 0.03)*(sys_pk(1,k) - dias_pk(1,k-shift_index)) + dias_pk(1,k-shift_index)))
                    lim_1=i;
                    break
                end
            end
   
            for i=sys_loc(1,k):dias_loc(1,k+1-shift_index)
                if(PPG(1,i) <= ((v(j) + 0.03)*(sys_pk(1,k) - dias_pk(1,k+1-shift_index)) + dias_pk(1,k+1-shift_index)))
                    lim_2=i;
                    break
                end
            end
            
%             scatter(lim_1,PPG(lim_1))
%             scatter(lim_2,PPG(lim_2))
            ppg_st(j) = (sys_loc(1,k) - lim_1)*Ts;
            ppg_dt(j) = (lim_2 - sys_loc(1,k))*Ts;
        end
        
        filerow_features = [];
        for i=1:length(v)
            filerow_features = [filerow_features ppg_dt(i) ppg_st(i) (ppg_st(i) + ppg_dt(i)) (ppg_dt(i)/ppg_st(i))];
        end
        
        filerow_features = [cp sys_time dias_time filerow_features];
        
        if(any(filerow_features <= 0) || any(isinf(filerow_features)))
            output_record = [];
            samples_deleted = samples_deleted + 1;
            break;
        end
        
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
        mbp = sbp/3 + abp*2/3;
        if(sbp >= 180 || abp >= 130 || sbp <= 80 || abp <= 60)
            continue
        end
        
        filerow_target = [mbp sbp abp];
        
        if(do_average >= 1 && do_average <= 7)
            output_record(end,:) = (output_record(end,:) + [(d*prt_number*10000) filerow_target filerow_features])/2;
            do_average = do_average + 1;
        else
            output_record = [output_record; (d*prt_number*10000) filerow_target filerow_features];
            do_average = 1;
        end
    end
%     close all;
    output_record = unique(output_record,'rows');
    output_file = [output_file; output_record];
    waitbar((d-1)/length(full_clean_record_indexes),f,'Extracting features...');
end 

writematrix(output_file,strcat('dataset_part',int2str(prt_number),'.csv'));
size(output_file)
close(f);
toc
disp('samples_deleted')
disp(samples_deleted)