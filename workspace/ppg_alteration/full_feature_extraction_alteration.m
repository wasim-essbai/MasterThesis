function full_feature_extraction_alteration(Part, alteration_type, alt_level, alt_index, prt_number)
y_test_ids = readmatrix('../data_split/y_test_ids.csv');

output_file=[];
skipped = 0;
index_skipped=[];
no_bp=0;

filerow_header = ["ID" "min1" "max1" "min2" "max2"];
filerow_header = [filerow_header "mbp" "sbp" "dbp"];
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
f = waitbar(0,strcat(int2str(alt_level),': Extracting features...'));
for d=1:length(y_test_ids)
    curr_index = y_test_ids(d,1) - prt_number * 10000;
    Y=Part{1,curr_index};
    signal_limit = min(size(Y(1,:),2),52500);
    PPG_original=Y(1,1:signal_limit);
    BP_original=Y(2,1:signal_limit);
    
    [b,a]=butter(4,[0.5*2*Ts,8*2*Ts]);
    PPG = filtfilt(b, a, PPG_original);
    
    % Alter signal
    PPG = apply_alteration(PPG, alteration_type, alt_level);
    
    % Signal normalization
    max_ppg = max(PPG);
    min_ppg = min(PPG);
    PPG = (PPG - min_ppg)/(max_ppg - min_ppg);
    
    [b,a]=butter(4,8*2*Ts);
    BP = filtfilt(b, a, BP_original);
    [BP,~,~,~] = hampel(BP,100,5);
    
    min1 = y_test_ids(d,2);
    max1 = y_test_ids(d,3);
    min2 = y_test_ids(d,4);
    max2 = y_test_ids(d,5);
  
    %% Feature extraction
    v = [0.1,0.25,0.33,0.5,0.66,0.75];

    ppg_st = zeros(1,length(v));
    ppg_dt = zeros(1,length(v));

    sys_time = (max1 - min1)*Ts;
    dias_time = (min2 - max1)*Ts;
    cp = (max2 - max1)*Ts;        

    if(sys_time <= 0 || dias_time <= 0 || cp <= 0)
        skipped = skipped + 1;
        continue
    end

    for j=1:length(v)
        lim_1 = 0;
        lim_2 = 0;
        for i=min1:max1
            if(PPG(1,i) >= ((v(j) - 0.03)*(PPG(1,max1) - PPG(1,min1)) + PPG(1,min1)))
                lim_1=i;
                break
            end
        end

        for i=max1:min2
            if(PPG(1,i) <= ((v(j) + 0.03)*(PPG(1,max1) - PPG(1,min2)) + PPG(1,min2)))
                lim_2=i;
                break
            end
        end

        ppg_st(j) = (max1 - lim_1)*Ts;
        ppg_dt(j) = (lim_2 - max1)*Ts;
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
        continue;
    end


    sbp = max(BP(1,max1:max2));
    abp = min(BP(1,max1:max2));
    mbp = sbp/3 + abp*2/3;

    filerow_identifier = [(curr_index+prt_number*10000) min1 max1 min2 max2];
    filerow_target = [mbp sbp abp];

    output_file = [output_file; filerow_identifier filerow_target filerow_features];
    waitbar((d-1)/length(y_test_ids),f,'Extracting features...');
end

writematrix(output_file,strcat('./altered_dataset/',alteration_type,'/dataset_part',int2str(prt_number),'_',alteration_type,'_',int2str(alt_index),'.csv'));
close(f);
toc
end