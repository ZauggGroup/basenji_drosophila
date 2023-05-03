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
mkdir /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/
/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/DGRP-28.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/DGRP-28/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/DGRP-28/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/DGRP-307.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/DGRP-307/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/DGRP-307/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/DGRP-399.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/DGRP-399/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/DGRP-399/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/DGRP-57.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/DGRP-57/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/DGRP-57/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/DGRP-639.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/DGRP-639/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/DGRP-639/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/DGRP-712.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/DGRP-712/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/DGRP-712/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/DGRP-714.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/DGRP-714/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/DGRP-714/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/DGRP-852.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/DGRP-852/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/DGRP-852/sequences.bed
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_predict_bed.py -f /g/furlong/project/103_Basenji/data/personalised_genomes/vgn.fa -o /g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/vgn/ --rc --shifts "1,0,-1" /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_chr_augmented/model_best.h5 /g/furlong/project/103_Basenji/data/personalised_predictions_32k/vgn/sequences.bed
conda deactivate
echo '...done.'