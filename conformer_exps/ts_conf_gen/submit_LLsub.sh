#!/bin/bash -l

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

# orca
source ~/RMG_shared/Software/orca/orca_env.sh

# gaussian
export g16root=/home/gridsan/groups/RMG/Software/gaussian
export PATH=$g16root/g16/:$g16root/gv:$PATH
export GAUSS_SCRDIR=/home/gridsan/lagnajit/scratch/$SLURM_JOB_NAME-$SLURM_JOB_ID-$LLSUB_RANK
export GAUSS_SCRDIR
. $g16root/g16/bsd/g16.profile
mkdir -p $GAUSS_SCRDIR
chmod 750 $GAUSS_SCRDIR

# script
# python generate_ts_confs_LLsub.py --exp_dir ./exps/test_methods --opt_method GFN2-xTB --rxns_path ~/code/ts_egnn/data/gsm_clean/wb97xd3.csv --split_path ~/code/ts_egnn/data/gsm_clean/splits/split0.pkl --task_id $LLSUB_RANK --num_tasks $LLSUB_SIZE --guess_method "ts_gcn"
python verify_ts_confs_LLsub.py --exp_dir ./exps/test_semiempirical_methods/gaussian/GFN2-xTB_test/ --opt_method "GFN2-xTB" --split_path ~/code/ts_egnn/data/gsm_clean/splits/split0.pkl --rxn_path ~/code/ts_egnn/data/gsm_clean/wb97xd3.csv --ts_path ~/code/ts_egnn/data/gsm_clean/wb97xd3_ts.sdf --task_id $LLSUB_RANK --num_tasks $LLSUB_SIZE

# cleanup
rm -rf $GAUSS_SCRDIR

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Elapsed time (s):" 
echo $ELAPSED_TIME
