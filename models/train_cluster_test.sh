#!/bin/bash
#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 50G
#SBATCH --time 1-00:00:00
#SBATCH -o /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.out
#SBAtCH -e /g/scb/zaugg/stojanov/basenji/experiments/cluster_outputs/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=frosina.stojanovska@embl.de
#SBATCH -p gpu
#SBATCH --qos=highest
module load cuDNN
export PATH="/g/scb/zaugg/stojanov/development/miniconda3/bin:$PATH"
echo 'activating virtual environment'
source activate /g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/
export PATH="/g/scb/zaugg/stojanov/basenji/bin:$PATH"
export PYTHONPATH="/g/scb/zaugg/stojanov/basenji:$PYTHONPATH"
echo '... done.'
echo 'starting training'
#cd /g/scb/zaugg/stojanov/basenji/experiments
cd /g/scb/zaugg/stojanov/basenji/bin
/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_train.py -o /g/scb/zaugg/stojanov/basenji/experiments/models/drosophila /g/scb/zaugg/stojanov/basenji/experiments/models/params_small.json /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l131k
conda deactivate
echo '...done.'