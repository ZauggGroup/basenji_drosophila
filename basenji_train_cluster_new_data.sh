#!/bin/bash
#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 50G
#SBATCH --time 7-00:00:00
#SBATCH -o /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.out
#SBAtCH -e /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frosina.stojanovska@embl.de
#SBATCH -p gpu-el8
#SBATCH -C gpu=A40
#SBATCH --qos=highest
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
export PATH="/g/scb/zaugg/stojanov/development/miniconda3/bin:$PATH"
echo 'activating virtual environment'
source activate /g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/
export PATH="/g/scb/zaugg/stojanov/basenji/bin:$PATH"
export PYTHONPATH="/g/scb/zaugg/stojanov/basenji:$PYTHONPATH"
echo '... done.'
echo 'starting training'
cd /g/scb/zaugg/stojanov/basenji/bin
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_new_data_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l131k.json /g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila_l131k
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_new_data_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l65k.json /g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila_l65k
/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_new_data_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l32k.json /g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila_l32k
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_chr_new_data_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l131k.json /g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila_l131k_chr
conda deactivate
echo '...done.'