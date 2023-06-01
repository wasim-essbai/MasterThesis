clc;
clear all;
close all; 
dataset = readmatrix('dataset_part2');
dataset_rest = dataset((dataset(:,3) < 120 & dataset(:,3) > 80 & dataset(:,4) < 80 & dataset(:,4) > 60),:);
writematrix(dataset_rest,strcat('dataset_part',int2str(2),'_rest.csv'));
test_rm = rmoutliers(dataset);
writematrix(test_rm,strcat('dataset_part',int2str(2),'_no_out.csv'));
test_rm_rest = test_rm((test_rm(:,3) <= 140 & test_rm(:,4) <= 100),:);
test_rm_rest = rmoutliers(test_rm_rest);
%writematrix(test_rm_rest,strcat('dataset_part',int2str(2),'_rest_no_out.csv'));

