#!/bin/bash
#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 28
#SBATCH --mem 100G
#SBATCH --time 1-00:00:00
#SBATCH -o /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.out
#SBAtCH -e /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frosina.stojanovska@embl.de
#SBATCH -p gpu-el8
#SBATCH -C gpu=A40
#SBATCH --gres=gpu:1
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
export PATH="/g/scb/zaugg/stojanov/development/miniconda3/bin:$PATH"
echo 'activating virtual environment'
source activate /g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/
export PATH="/g/scb/zaugg/stojanov/basenji/bin:$PATH"
export PYTHONPATH="/g/scb/zaugg/stojanov/basenji:$PYTHONPATH"
echo '... done.'
echo 'starting testing'
cd /g/scb/zaugg/stojanov/basenji/bin
/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/reference/dm6.UCSC.noMask.fa -o /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/reference/preds/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/reference/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/vgn/vgn.fa -o /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/vgn/preds/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/vgn/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/dgp_28/DGRP-28.fa -o /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/dgp_28/preds/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/personalized_predictions/dgp_28/sequences.bed
conda deactivate
echo '...done.'