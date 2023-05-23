clc;
clear all;
close all; 
dataset  = readtable('dataset_part1.csv');
X = dataset{:,:}(1:end,4:end);
y = dataset{:,:}(1:end,2);
mdl = fitlm(X,y);