#!/bin/bash

# Define lists of values for parameters
m_values=(3 5)
loss_values=(0)
opt_values=(4)
b_values=(32)
noe_values=(0 1)
rgb_values=(1)
lr_values=(0.25)
e_values=(20)
s_values=(1)
t_values=(0.5)
test_values=(0)
split_values=(0.85)
aug_values=(0)
verbose=0
cache=1
idv="final3"
mode="train"

# Loop over all possible combinations of values and call my_script.py
for aug in "${aug_values[@]}"; do # Loop load unlabeled image options
for noe in "${noe_values[@]}"; do # Loop load unlabeled image options
for s in "${s_values[@]}"; do # Loop Dataset size
for split in "${split_values[@]}"; do # Loop train split sizes
for rgb in "${rgb_values[@]}"; do  # Loop RGB or Grayscale
for e in "${e_values[@]}"; do # Loop Epochs
for t in "${t_values[@]}"; do # Loop Thresholds Value for Correctness
for test_val in "${test_values[@]}"; do # Loop Testset values
for lr in "${lr_values[@]}"; do # Loop Learning Rate
for opt in "${opt_values[@]}"; do  # Loop Optimizer
for loss in "${loss_values[@]}"; do   # Loop Loss Function
for b in "${b_values[@]}"; do   # Loop Batch Size
for m in "${m_values[@]}"; do # Loop Model

  sbatch run_batch.sh -mode "$mode" -aug "$aug" -id "$idv" -test "$test_val" -m "$m" -opt "$opt" -loss "$loss" -rgb "$rgb" -noe "$noe" -b "$b" -lr "$lr" -e "$e" -t "$t" -s "$s" -v "$verbose" -cache "$cache" -split "$split"

  # This is useful to keep track the parameters that are being used in the queued jobs
  touch out/m_"$m"-opt_"$opt"-loss_"$loss"-rgb_"$rgb"-noe_"$noe"-b_"$b"-lr_"$lr"-e_"$e"-t_"$t"-s_"$s"-split_"$split"-mode_"$mode"-aug_"$aug"-id_"$idv"-test_"$test_val".txt

done
done
done
done
done
done
done
done
done
done
done
done
done