#!/bin/bash
#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem 300G
#SBATCH --time 7-00:00:00
#SBATCH -o /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.out
#SBAtCH -e /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frosina.stojanovska@embl.de
#SBATCH -p gpu-el8
#SBATCH -C gpu=A100
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
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l131k.json /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l131k
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l65k.json /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l65k
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l32k.json /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l32k
/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_chr_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l131k.json /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l131k_chr
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_chr_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l65k.json /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l65k_chr
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l32k.json /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l32k_chr
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k /g/scb/zaugg/stojanov/basenji/experiments/models/params_drosophila_l65k_no_aug.json /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l65k
conda deactivate
echo '...done.'