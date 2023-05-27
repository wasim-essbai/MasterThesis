clc;
clear all;
close all; 
prt_number = 2;
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(prt_number)));

output_file=[];
samples_deleted = 0;
bad_peaks = 0;

filerow_header = ["ID" "sbp" "dbp"];
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
for d=1:length(Part_2)
    Y=Part_2{1,d};
    PPG_original=Y(1,300:800);
    BP_original=Y(2,300:800);
    
%     figure('Name','PPG and BP');
%     subplot(2,1,1);
%     plot(PPG_original);
% 
%     subplot(2,1,2);
%     plot(BP_original);

  
    % Filtering    
    %windowSize = 15;
    %PPG = filter((1/windowSize)*ones(1,windowSize),1,PPG_original);
    %[b,a]=butter(3,[0.5*2*Ts,8*2*Ts]); % Bandpass digital filter design 
    %PPG = filtfilt(b, a, PPG_original);
    
    %[b,a]=butter(1,8*2*Ts); % Bandpass digital filter design 
    %BP = filtfilt(b, a, BP_original);
    
    PPG = PPG_original;
    BP = BP_original;
    
    T =(0:Ts:(length(PPG)-1)*Ts); %time vector based on sampling rate
    
    [sys_pk,sys_loc]= findpeaks(PPG);
    [dias_pk,dias_loc]=findpeaks(-PPG);
    dias_pk = -dias_pk;
    
    if(any(sys_pk < mean(sys_pk)/4))
        output_record = [];
        samples_deleted = samples_deleted + 1;
        continue;
    end
    
%     figure('Name','PPG');
%     plot(PPG(1,1:end));
%     hold on
%     scatter(sys_loc, sys_pk)
%     scatter(dias_loc, dias_pk)
%     hold off
    
%     [sys_bp_pk,sys_bp_loc]=findpeaks(BP); 
%     [dias_bp_pk,dias_bp_loc]=findpeaks(-BP); % min value of BP(diastole) signal    
%     dias_bp_pk = -dias_bp_pk;
% 
%     figure('Name','BP');
%     plot(BP);
%     hold on
%     scatter(sys_bp_loc, sys_bp_pk)
%     scatter(dias_bp_loc, dias_bp_pk)
%     hold off

    %figure('Name','PPG and BP');
    %subplot(2,1,1);
    %plot(PPG);

    %subplot(2,1,2);
    %plot(BP);
    %xline(dias_loc(1));
    %xline(dias_loc(2));
    
    shift_index = 0;
    if(sys_loc(1,1) < dias_loc(1,1))
        shift_index=1;
    end
    
    output_record = [];
    

    last_index = min([length(sys_loc)-1, length(dias_loc)-1]);
    for k=(1+shift_index):(1+shift_index)
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
        
        sbp = max(BP(1,sys_loc(1,k):sys_loc(1,k+1)));
        abp = min(BP(1,sys_loc(1,k):sys_loc(1,k+1)));
        filerow_target = [sbp abp];
        
        %sbp >= 180 || abp >= 130 || sbp <= 80
        if(any(filerow_features <= 0) || any(isinf(filerow_features)) || sbp >= 180 || abp >= 130)
            output_record = [];
            samples_deleted = samples_deleted + 1;
            break;
        end
        
        output_record = [output_record; (d*prt_number*10000) filerow_target filerow_features];
        

        %if(sbp < 180 && abp < 130 && sbp > 80)
            %output_record = [output_record; d filerow_target filerow_features];
        %else
            %continue
%             figure('Name','PPG and BP');
% 
%             subplot(2,1,1);
%             plot(PPG);
%             hold on
%             scatter(sys_loc, sys_pk)
%             scatter(dias_loc, dias_pk)
%             hold off
% 
%             subplot(2,1,2);
%             plot(BP);
%             hold on
%             scatter(sys_loc, BP(sys_loc))
%             scatter(dias_loc, BP(dias_loc))
%             hold off
        %end
    end
    
    output_file = [output_file; output_record];
    waitbar((d-1)/3000,f,'Extracting features...');
end 


writematrix(output_file,strcat('dataset_part',int2str(prt_number),'.csv'));
%writematrix(output_file,'dataset_part1.csv','WriteMode','append');
close(f);
toc
disp('samples_deleted')
disp(samples_deleted)