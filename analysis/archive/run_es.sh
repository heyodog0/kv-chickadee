#!/bin/bash
#SBATCH -p gershman
#SBATCH -c 4
#SBATCH -t 0-00:45
#SBATCH --mem=24G
#SBATCH -J chick_es
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/es.log
#SBATCH -e /n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode/es.log
cd /n/netscratch/gershman_lab/Everyone/truong/chickadee_barcode
./.venv/bin/python -u event_specificity.py
echo "=== SLURM JOB DONE rc=$? ==="
