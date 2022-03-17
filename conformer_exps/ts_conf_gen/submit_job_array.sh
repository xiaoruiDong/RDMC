#!/bin/bash -l
#SBATCH -p xeon-p8
#SBATCH -J ml_conf_gen_gm
#SBATCH -c 20
#SBATCH --time=0-99:00:00
#######SBATCH --mem=64000 				# memory pool for all cores
######SBATCH --mem-per-cpu=1972
#SBATCH -e err-%A-%a.txt
#SBATCH -o out-%A-%a.txt
#SBATCH --array=0-1191

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "memory per node : $SLURM_MEM_PER_NODE"
echo "============================================================"

START_TIME=$SECONDS

source /etc/profile
module load anaconda/2020b
source activate rdmc_env

python generate_ts_confs.py --rxn_idx $SLURM_ARRAY_TASK_ID --exp_dir ./runs/run1/ --rxns_path val_rxns.csv

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Elapsed time (s):" 
echo $ELAPSED_TIME
