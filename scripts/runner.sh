#!/bin/bash
#SBATCH -n 1
#SBATCH -N 10
#SBATCH -t 0-00:02
python scripts/test_NISTfit-LM.py 16
python scripts/test_NISTfit-evaluation.py 16
