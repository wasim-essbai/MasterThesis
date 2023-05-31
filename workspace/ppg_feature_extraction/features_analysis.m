clc;
clear all;
close all; 
dataset  = readtable('dataset_part1.csv');

sbp = dataset{:,:}(1:end,2);
dbp = dataset{:,:}(1:end,3);

plot(dataset{:,:}(1:end,5))

