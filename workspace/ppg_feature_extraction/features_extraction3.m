clc;
clear all;
close all; 
prt_number = 4;
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(prt_number)));
load(strcat('clean_record_indexes_',int2str(prt_number)));

output_file=[];
samples_deleted = 0;
bad_peaks = 0;
added=[];

filerow_header = ["ID" "mbp" "sbp" "dbp"];
filerow_header = [filerow_header "amp" "cp" "sut" "dt"];
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
for d=2:length(clean_record_indexes)
    Y=Part_4{1,clean_record_indexes(d)};
    PPG_original=Y(1,length(Y)/2-499:length(Y)/2+500);
    BP_original=Y(2, length(Y)/2-499:length(Y)/2+500);
  
    [b,a]=butter(4,[0.5*2*Ts,8*2*Ts]);
    PPG = filtfilt(b, a, PPG_original);
    PPG = PPG(200:800);
   
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence',max(PPG)/8);
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence',max(PPG)/8);
    dias_pk = -dias_pk;
    
    % Signal normalization
    max_ppg = max(PPG);
    min_ppg = min(PPG);
    PPG = (PPG - min_ppg)/(max_ppg - min_ppg);
    sys_pk = (sys_pk - min_ppg)/(max_ppg - min_ppg);
    dias_pk = (dias_pk - min_ppg)/(max_ppg - min_ppg);
    
    [PPG_d]=gradient(PPG);
    %figure('Name','PPG 1st derivative');

    %plot(Fy);
    [PPG_dd]=gradient(PPG_d);  % 2nd derivative
    
    [max_pk_dd, max_loc_dd]= findpeaks(PPG_dd);
    [min_pk_dd, min_loc_dd]=findpeaks(-PPG_dd);
    min_pk_dd = -min_pk_dd;
    
    plot(PPG_dd)
    hold on
    scatter(max_loc_dd, max_pk_dd)
    scatter(min_loc_dd, min_pk_dd)
    hold off
    
    [b,a]=butter(4,8*2*Ts);
    BP = filtfilt(b, a, BP_original);
    [BP,i,bpmedian,bpsigma] = hampel(BP);
    BP = BP(200:800);
    
    [sys_bp_pk,sys_bp_loc]=findpeaks(BP, 'MinPeakProminence', max(BP)/8); 
    [dias_bp_pk,dias_bp_loc]=findpeaks(-BP, 'MinPeakProminence', max(BP)/8);
    dias_bp_pk = -dias_bp_pk;
    
    shift_index = 0;
    if(sys_loc(1,1) < dias_loc(1,1))
        shift_index=1;
    end    
    
    output_record = [];
    last_index = min([length(sys_loc)-1, length(dias_loc)]);
%     figure
%     plot(PPG)
%     hold on
%     scatter(sys_loc,sys_pk)
%     scatter(dias_loc,dias_pk)
    for k=(1+shift_index):last_index
        sys_time = (sys_loc(1,k) - dias_loc(1,k-shift_index))*Ts;
        dias_time = (dias_loc(1,k+1-shift_index) - sys_loc(1,k))*Ts;
        cp = (sys_loc(1,k+1) - sys_loc(1,k))*Ts;
        amp = sys_pk(1,k) - dias_pk(1,k-shift_index);

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
        
        max_second_der = max_loc_dd(max_loc_dd >= dias_pk(1,k-shift_index) && max_loc_dd <= dias_pk(1,k+1-shift_index));
        min_second_der = min_loc_dd(min_loc_dd >= dias_pk(1,k-shift_index) && min_loc_dd <= dias_pk(1,k+1-shift_index));
        a = max_second_der(1);
        b = max_second_der(1);
        filerow_features = [];
        for i=1:length(v)
            filerow_features = [filerow_features ppg_dt(i) ppg_st(i) ppg_st(i) + ppg_dt(i) ppg_dt(i)/ppg_st(i)];
        end
        
        filerow_features = [amp cp sys_time dias_time filerow_features];
        
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
        filerow_target = [mbp sbp abp];
        
        output_record = [output_record; (d*prt_number*10000) filerow_target filerow_features];
    end
%     close all;
    output_file = [output_file; output_record];
    waitbar((d-1)/length(clean_record_indexes),f,'Extracting features...');
end 

writematrix(output_file,strcat('dataset_part',int2str(prt_number),'.csv'));
size(output_file)
close(f);
toc
disp('samples_deleted')
disp(samples_deleted)