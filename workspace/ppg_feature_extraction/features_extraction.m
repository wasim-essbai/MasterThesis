clc;
clear all;
close all; 
load('F:/Università/Magistrale/Tesi/workspace/dataset/part_2');

step = 1;
h=0;
output_file=[];

if(step==1)
    filerow_header = ["sbp" "dbp"];
    filerow_header = [filerow_header "cp" "sut" "dt"];
    filerow_header = [filerow_header "dt10" "st10_p_dt10" "st10_d_dt10"];
    filerow_header = [filerow_header "dt25" "st25_p_dt25" "st25_d_dt25"];
    filerow_header = [filerow_header "dt33" "st33_p_dt33" "st33_d_dt33"];
    filerow_header = [filerow_header "dt50" "st50_p_dt50" "st50_d_dt50"];
    filerow_header = [filerow_header "dt66" "st66_p_dt66" "st66_d_dt66"];
    filerow_header = [filerow_header "dt75" "st75_p_dt75" "st75_d_dt75"];
    output_file = [output_file;filerow_header];
end

Ts=1/125;

sample_length = 1000;
start_index = (step-1)*sample_length + 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
f = waitbar(0,'Extracting features...');
for d=start_index:(step*sample_length)
    Y=p{1,d};
    PPG=Y(1,1:1000);
    BP=Y(2,1:1000);

    %figure('Name','PPG and BP');
    %subplot(2,1,1);
    %plot(PPG);

    %subplot(2,1,2);
    %plot(BP);

    T =(0:Ts:(length(PPG)-1)*Ts); %time vector based on sampling rate
    
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence',max(PPG)/11); % max values of PPG signal
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence',max(PPG)/11,'MinPeakDistance',25); % min values of PPG signal
    dias_pk = -dias_pk;
    
    
%    figure('Name','PPG');
%    plot(PPG(1,1:end));
%    hold on
%    scatter(sys_loc, sys_pk)
%    scatter(dias_loc, dias_pk)
%    hold off
    
    [sys_bp_pk,sys_bp_loc]=findpeaks(BP); 
    [dias_bp_pk,dias_bp_loc]=findpeaks(-BP); % min value of BP(diastole) signal    
    dias_bp_pk = -dias_bp_pk;

    %figure('Name','BP');
    %plot(BP);
    %hold on
    %scatter(sys_bp_loc, sys_bp_pk)
    %scatter(dias_bp_loc, dias_bp_pk)
    %hold off

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
    
    last_index = min([length(sys_pk)-1, length(dias_loc)-1, length(sys_bp_loc), length(dias_bp_loc)]);
    for k=(1+shift_index):last_index
        sys_time = T(sys_loc(1,k))-T(dias_loc(1,k-shift_index));
        dias_time = T(dias_loc(1,k+1-shift_index))-T(sys_loc(1,k));
        cp = T(sys_loc(1,k+1))-T(sys_loc(1,k));

        v = [0.1,0.25,0.33,0.5,0.66,0.75];

        ppg_st = zeros(1,length(v));
        ppg_dt = zeros(1,length(v));
        for j=1:length(v)
            for i=dias_loc(1,k-shift_index):sys_loc(1,k)
                if(PPG(1,i) >= (v(j)*(sys_pk(1,k) - dias_pk(1,k-shift_index)) + dias_pk(1,k-shift_index)))
                    a=i;
                    break
                end
            end
   
            for i=sys_loc(1,k):dias_loc(1,k+1-shift_index)
                if(PPG(1,i) <= (v(j)*(sys_pk(1,k) - dias_pk(1,k+1-shift_index)) + dias_pk(1,k+1-shift_index)))
                    b=i;
                    break
                end
            end

            ppg_st(j) = (sys_loc(1,k)-a)*Ts;
            ppg_dt(j) = (b-sys_loc(1,k))*Ts;
        end
        filerow_features = [];
        for i=1:length(v)
            filerow_features = [filerow_features ppg_dt(i) ppg_st(i)+ppg_dt(i) ppg_dt(i)/ppg_st(i)];
        end
        filerow_features = [cp sys_time dias_time filerow_features];
        
        if(any(filerow_features < 0) || any(filerow_features == 0) || any(isinf(filerow_features)))
            continue
        end
        filerow_target= [sys_bp_pk(1,k) dias_bp_pk(1,k)];
        h = h + 1;
        output_file = [output_file;filerow_target filerow_features];
    end
    waitbar((d-start_index)/sample_length,f,'Extracting features...');
end 

writematrix(output_file,'dataset_part2.csv');
%writematrix(output_file,'dataset_part1.csv','WriteMode','append');
close(f);
toc