clc;
clear all;
close all; 
dataset  = readtable('dataset_part1.csv');

sbp = dataset{:,:}(1:end,2);
plot(sbp)
dbp = dataset{:,:}(1:end,3);
plot(dbp)

sum(sbp>=180)
sum(dbp>=130)

sum(sbp>=180 & dbp>=130)

