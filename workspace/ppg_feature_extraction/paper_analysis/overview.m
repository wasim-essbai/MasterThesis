clc;
clear all;
close all; 

for check_no = 44:4010
    sample = readmatrix(strcat('./cleaned/cleaned/train/check',int2str(check_no),'.csv'));
    figure
    subplot(2,1,1);
    plot(sample(1:end,1));

    subplot(2,1,2);
    plot(sample(1:end,2));
    close all;
end

bp_train_new = readmatrix('./cleaned/bp_train_new.csv');