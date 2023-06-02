function full_feature_extraction_alteration(alteration_type, alt_level)
load(strcat('F:/Università/Magistrale/Tesi/workspace/dataset/part_',int2str(2)));
y_test_ids = readmatrix('../bp_estimation_ANN/data_split/y_test_ids.csv');
y_test_ids = unique(y_test_ids);

for d=1:length(y_test_ids)
    curr_index = floor(y_test_ids(d)/10000);
    Y=Part_2{1,curr_index};
    signal_limit = min(size(Y(1,:),2),52500);
    PPG_original=Y(1,1:signal_limit);
    BP_original=Y(2,1:signal_limit);
    
    [b,a]=butter(4,[0.5*2*Ts,8*2*Ts]);
    PPG = filtfilt(b, a, PPG_original);
    [~,sys_loc]= findpeaks(PPG,'MinPeakProminence',max(PPG)/6);
    [~,dias_loc]=findpeaks(-PPG,'MinPeakProminence',max(PPG)/8);
    
    mean_sys_dist = 0;
    for j=2:length(sys_loc)
        mean_sys_dist = mean_sys_dist + sys_loc(1,j) - sys_loc(1,j-1);
    end
    mean_sys_dist = round((mean_sys_dist/length(sys_loc))/2);
    
    mean_dias_dist = 0;
    for j=2:length(dias_loc)
        mean_dias_dist = mean_dias_dist + dias_loc(1,j) - dias_loc(1,j-1);
    end
    mean_dias_dist = round((mean_dias_dist/length(dias_loc))/2);
    
    [sys_pk,sys_loc]= findpeaks(PPG,'MinPeakProminence',max(PPG)/6,'MinPeakDistance', mean_sys_dist);
    [dias_pk,dias_loc]=findpeaks(-PPG,'MinPeakProminence',max(PPG)/6,'MinPeakDistance', mean_dias_dist);
    dias_pk = -dias_pk;
    
    % Signal normalization
    max_ppg = max(PPG);
    min_ppg = min(PPG);
    PPG = (PPG - min_ppg)/(max_ppg - min_ppg);
    sys_pk = (sys_pk - min_ppg)/(max_ppg - min_ppg);
    dias_pk = (dias_pk - min_ppg)/(max_ppg - min_ppg);
    
    [b,a]=butter(4,8*2*Ts);
    BP = filtfilt(b, a, BP_original);
    [BP,~,~,~] = hampel(BP,100,5);
    
    [sys_bp_pk,sys_bp_loc]=findpeaks(BP, 'MinPeakProminence', max(BP)/8); 
    [dias_bp_pk,dias_bp_loc]=findpeaks(-BP, 'MinPeakProminence', max(BP)/8);
    dias_bp_pk = -dias_bp_pk;
    
        shift_index = 0;
    if(sys_loc(1,1) < dias_loc(1,1))
        shift_index=1;
    end    
    
    sys_loc(sys_pk < mean(sys_pk)/1.6)=[];
    sys_pk(sys_pk < mean(sys_pk)/1.6)=[];
    
    dias_loc(dias_pk > mean(dias_pk)*2.5)=[];
    dias_pk(dias_pk > mean(dias_pk)*2.5)=[];
    
    % Alter signal
    PPG = apply_alteration(PPG, alteration_type, alt_level);
    
    output_record = [];
    last_index = min([length(sys_loc)-1, length(dias_loc)-1]);
    do_average = 0;
    last_dias_index=0;
    for k=(2+shift_index):last_index
        v = [0.1,0.25,0.33,0.5,0.66,0.75];
        
        ppg_st = zeros(1,length(v));
        ppg_dt = zeros(1,length(v));
        dias_index = length(dias_loc(1,dias_loc < sys_loc(1,k)));
        if(dias_index <= 0 || last_dias_index == dias_index)
            skipped = skipped + 1;
            index_skipped = [index_skipped d];
            continue
        end
        last_dias_index = dias_index;
        
        if(sys_loc(1,k) <= dias_loc(1,dias_index) ...
            || sys_loc(1,k) >= dias_loc(1,dias_index+1) ...
            || sys_loc(1,k+1) <= dias_loc(1,dias_index+1) ...
            || sys_loc(1,k-1) >= dias_loc(1,dias_index) ...
            || (sys_loc(1,k)-dias_loc(1,dias_index)) >= (mean_dias_dist + mean_sys_dist)*1.3 ...
            || (dias_loc(1,dias_index+1)-sys_loc(1,k)) >= (mean_dias_dist + mean_sys_dist)*1.3 ... 
            || (sys_loc(1,k+1) - sys_loc(1,k)) >= (mean_sys_dist*1.3*2))
            skipped = skipped + 1;
            index_skipped = [index_skipped d];
            continue
        end
        
        sys_time = (sys_loc(1,k) - dias_loc(1,dias_index))*Ts;
        dias_time = (dias_loc(1,dias_index+1) - sys_loc(1,k))*Ts;
        cp = (sys_loc(1,k+1) - sys_loc(1,k))*Ts;        

        if(sys_time <= 0 || dias_time <= 0 || cp <= 0)
            skipped = skipped + 1;
            index_skipped = [index_skipped d];
            continue
        end
        
        for j=1:length(v)
            lim_1 = 0;
            lim_2 = 0;
            for i=dias_loc(1,dias_index):sys_loc(1,k)
                if(PPG(1,i) >= ((v(j) - 0.03)*(sys_pk(1,k) - dias_pk(1,dias_index)) + dias_pk(1,dias_index)))
                    lim_1=i;
                    break
                end
            end
   
            for i=sys_loc(1,k):dias_loc(1,dias_index+1)
                if(PPG(1,i) <= ((v(j) + 0.03)*(sys_pk(1,k) - dias_pk(1,dias_index+1)) + dias_pk(1,dias_index+1)))
                    lim_2=i;
                    break
                end
            end
            
            ppg_st(j) = (sys_loc(1,k) - lim_1)*Ts;
            ppg_dt(j) = (lim_2 - sys_loc(1,k))*Ts;
        end
        
        if(any(ppg_st <= 0) || any(ppg_dt <= 0))
            skipped = skipped + 1;
            index_skipped = [index_skipped d];
            continue
        end
        
        filerow_features = [];
        for i=1:length(v)
            filerow_features = [filerow_features ppg_dt(i) ppg_st(i) (ppg_st(i) + ppg_dt(i)) (ppg_dt(i)/ppg_st(i))];
        end
        
        filerow_features = [cp sys_time dias_time filerow_features];
        
        if(any(filerow_features <= 0) || any(isinf(filerow_features)))
            skipped = skipped + 1;
            index_skipped = [index_skipped d];
            continue;
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
        
        if((sys_bp_presence+dias_bp_presence) < 2)
            no_bp = no_bp+1;
            no_bp=no_bp+1;
            continue;
        end
        
        sbp = max(BP(1,sys_loc(1,k):sys_loc(1,k+1)));
        abp = min(BP(1,sys_loc(1,k):sys_loc(1,k+1)));
        mbp = sbp/3 + abp*2/3;
%         if(sbp >= 180 || abp >= 130 || sbp <= 80 || abp <= 60)
%             no_bp=no_bp+1;
%             continue
%         end
        
        filerow_target = [mbp sbp abp];
        
        if(do_average >= 1 && do_average <= 4)
            output_record(end,:) = (output_record(end,:) + [(d*prt_number*10000) filerow_target filerow_features])/2;
            do_average = do_average + 1;
        else
            output_record = [output_record; (d*prt_number*10000) filerow_target filerow_features];
            do_average = 1;
        end
    end

    output_record = unique(output_record,'rows');
    output_file = [output_file; output_record];
    waitbar((d-1)/length(full_clean_record_indexes),f,'Extracting features...');
end
end