clc;
clear all;
close all; 
prt_number = 3;
load(strcat('full_first_step_clean_record_indexes_',int2str(prt_number),'.mat'));
load(strcat('clean_record_indexes_',int2str(prt_number),'.mat'));

full_clean_record_indexes = intersect(first_step_clean_record_indexes, clean_record_indexes);

save(strcat('full_clean_record_indexes',int2str(prt_number),'.mat'),'full_clean_record_indexes');