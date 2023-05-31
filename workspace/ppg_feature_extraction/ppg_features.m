tic
clc;
%clear all;
close all; 
load('Part_4');
FILE=[];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for d=1:3000
Y=(Part_4{1,d});
O1P=Y(1,1:1000);
BP=Y(2,1:1000);
O1E=Y(3,1:1000);

Ts=1/125; 
T =(0:0.008:7.999); 

[pk,loc]= findpeaks(O1P); % max value of PPG signal
PPG1=max(O1P)-O1P; % To find out the min peak of PPG
[pk1,loc1]=findpeaks(PPG1,'MinPeakHeight',0.0); % min value of PPG signal

sys_time=0;
for i=1:1:5
    sys_time = sys_time + T(loc(1,i))-T(loc1(1,i));
end
sys_time = sys_time/5;

dias_time=0;
for i=1:1:5
    dias_time = dias_time + T(loc1(1,i+1))-T(loc(1,i));
end


dias_time = dias_time/5;

v = [0.1,0.25,0.33,0.5,0.66,0.75];

ppg_21_st = [];
ppg_21_dt = [];
for j=1:1:6
for i=loc1(1,1):1:loc(1,1)
    if(O1P(1,i)>=(v(j)*pk(1,1)+pk1(1,1)))
        a=i;
   
        break
    end
end

for i=loc(1,1):1:loc1(1,2)
    if(O1P(1,i)<=(v(j)*pk(1,1)+ pk1(1,1)))
        b=i;
        
        break
    end
end

ppg_21_st(j) = (loc(1,1)-a)*0.008;
ppg_21_dt(j) = (b-loc(1,1))*0.008;

end


        
%findpeaks(PPG1,'MinPeakHeight',0.6);  % noise threshold
%figure('Name','min PPG after threshold');

%plot(PPG1);

[pk5,loc5]=findpeaks(BP); 


%findpeaks(BP);
BP1=max(BP)-BP; % To find out the min peak of BP
[pk6,loc6]=findpeaks(BP1); % min value of BP(diastole) signal    
%findpeaks(BP1);


[lr,lr1] = size(loc5);
bpmax=0;
for i=1:1:lr1-1
    
    bpmax = bpmax + pk5(1,i);
end

bpmax = bpmax/(lr1-1);

[lr,lr1] = size (loc6);
bpmin=0;
for i=1:1:lr1-1
    bpmin = bpmin + pk6(1,i);
end

bpmin = bpmin/(lr1-1);


filerow1 = [ppg_21_dt(1) ppg_21_st(1)+ppg_21_dt(1) ppg_21_dt(1)/ppg_21_st(1) ppg_21_dt(2) ppg_21_st(2)+ppg_21_dt(2) ppg_21_dt(2)/ppg_21_st(2) ppg_21_dt(3) ppg_21_st(3)+ppg_21_dt(3) ppg_21_dt(3)/ppg_21_st(3) ppg_21_dt(4) ppg_21_st(4)+ppg_21_dt(4) ppg_21_dt(4)/ppg_21_st(4) ppg_21_dt(5) ppg_21_st(5)+ppg_21_dt(5) ppg_21_dt(5)/ppg_21_st(5) ppg_21_dt(6) ppg_21_st(6)+ppg_21_dt(6) ppg_21_dt(6)/ppg_21_st(6) sys_time dias_time];
filerow = [real(bpmax) real(bpmin)];
 
%%%change here to include filerow
FILE = [FILE;filerow1];
 
end
csvwrite('ppg4.csv',FILE);
toc