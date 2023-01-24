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
#module load CUDA/11.3.1
#module load cuDNN/8.2.1.32-CUDA-11.3.1
export PATH="/g/scb/zaugg/stojanov/development/miniconda3/bin:$PATH"
export PATH="/g/scb/zaugg/stojanov/development/miniconda3/bin:$PATH"
echo 'activating virtual environment'
source activate /g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/
export PATH="/g/scb/zaugg/stojanov/basenji/bin:$PATH"
export PATH=/home/stojanov/meme/bin:/home/stojanov/meme/libexec/meme-5.4.1:$PATH
export PYTHONPATH="/g/scb/zaugg/stojanov/basenji:$PYTHONPATH"
echo '... done.'
echo 'starting testing'
#cd /g/scb/zaugg/stojanov/basenji/experiments
cd /g/scb/zaugg/stojanov/basenji/bin
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_motifs.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented/motifs -m /g/scb/zaugg/stojanov/basenji/experiments/data/Drosophila_melanogaster.meme -t -d /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l131k
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_motifs.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented/motifs_selected -m /g/scb/zaugg/stojanov/basenji/experiments/data/selected_motifs.meme -t -d /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l131k
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_motifs.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_augmented/motifs -m /g/scb/zaugg/stojanov/basenji/experiments/data/Drosophila_melanogaster.meme -t -d /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l65k
/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_motifs.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_augmented/motifs_selected -m /g/scb/zaugg/stojanov/basenji/experiments/data/selected_motifs.meme -t -d /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l65k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l65k
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_motifs.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/motifs -m /g/scb/zaugg/stojanov/basenji/experiments/data/Drosophila_melanogaster.meme -t -d /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l32k
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_motifs.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/motifs_selected -m /g/scb/zaugg/stojanov/basenji/experiments/data/selected_motifs.meme -t -d /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l32k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l32k
conda deactivate
echo '...done.'