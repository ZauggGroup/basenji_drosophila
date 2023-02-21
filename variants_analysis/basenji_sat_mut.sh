#!/bin/bash
#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 28
#SBATCH --mem 500G
#SBATCH --time 7-00:00:00
#SBATCH -o /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.out
#SBAtCH -e /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frosina.stojanovska@embl.de
#SBATCH -p gpu-el8
#SBATCH -C gpu=A40
#SBATCH --gres=gpu:1
#SBATCH --qos=high
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
/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_sat_vcf.py --f1 /g/scb/zaugg/stojanov/basenji/experiments/data/dm6.UCSC.noMask.fa -l 200 -o /g/scb/zaugg/stojanov/basenji/experiments/variants_predictions/sat_mut/ --rc --shifts "1,0,-1" -t /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila.txt /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented/params.json /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented/model_best.h5 /g/scb/zaugg/stojanov/basenji/experiments/data/Haplotype_joint_call_F1_stringent_filtering.vcf
conda deactivate
echo '...done.'
