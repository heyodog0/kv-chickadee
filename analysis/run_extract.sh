#!/bin/bash
#SBATCH -p gershman
#SBATCH -c 4
#SBATCH -t 0-01:30
#SBATCH --mem=48G
#SBATCH -J chick_extract
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/extract.log
#SBATCH -e /n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/extract.log
module load matlab/R2024b-fasrc01
cd /n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode
matlab -nodisplay -nosplash -batch "extract_events"
echo "=== SLURM JOB DONE rc=$? ==="
