#!/bin/bash
#SBATCH -n 1
#SBATCH -N 10
#SBATCH -t 0-00:04
#SBATCH --output=/home/ihb/Code/NISTfit
echo $(hostname)
cat /proc/cpuinfo
python scripts/time_NISTfit.py 16
