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
#SBATCH --qos=highest
export PATH="/g/scb/zaugg/stojanov/development/miniconda3/bin:$PATH"
echo 'activating virtual environment'
source activate /g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/
export PATH="/g/scb/zaugg/stojanov/basenji/bin:$PATH"
export PYTHONPATH="/g/scb/zaugg/stojanov/basenji:$PYTHONPATH"
echo '... done.'
echo 'starting'
cd /g/scb/zaugg/stojanov/basenji/bin
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_data.py -l 131072 --local -o /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l131k -p 8 -t .1 -v .1 -w 128 -c 8192 -u /g/scb/zaugg/stojanov/basenji/experiments/data/unmap.bed /g/scb/zaugg/stojanov/basenji/experiments/data/dm6.UCSC.noMask.fa /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila.txt
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_data.py -l 65536 --local -o /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l65k -p 8 -t .1 -v .1 -w 128 -c 4096 -u /g/scb/zaugg/stojanov/basenji/experiments/data/unmap.bed /g/scb/zaugg/stojanov/basenji/experiments/data/dm6.UCSC.noMask.fa /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila.txt
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_data.py -l 32768 --local -o /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l32k -p 8 -t .1 -v .1 -w 128 -c 2048 -u /g/scb/zaugg/stojanov/basenji/experiments/data/unmap.bed /g/scb/zaugg/stojanov/basenji/experiments/data/dm6.UCSC.noMask.fa /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila.txt
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_data.py -l 131072 --local -o /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l131k_chr -p 8 -t chr2R -v chr3L -w 128 -c 8192 -u /g/scb/zaugg/stojanov/basenji/experiments/data/unmap.bed /g/scb/zaugg/stojanov/basenji/experiments/data/dm6.UCSC.noMask.fa /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila.txt
#/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_data.py -l 65536 --local -o /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l65k_chr -p 8 -t chr2R -v chr3L -w 128 -c 4096 -u /g/scb/zaugg/stojanov/basenji/experiments/data/unmap.bed /g/scb/zaugg/stojanov/basenji/experiments/data/dm6.UCSC.noMask.fa /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila.txt
/g/scb/zaugg/stojanov/development/miniconda3/envs/basenji/bin/python basenji_data.py -l 32768 --local -o /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l32k_chr -p 8 -t chr2R -v chr3L -w 128 -c 2048 -u /g/scb/zaugg/stojanov/basenji/experiments/data/unmap.bed /g/scb/zaugg/stojanov/basenji/experiments/data/dm6.UCSC.noMask.fa /g/scb/zaugg/stojanov/basenji/experiments/data/drosophila.txt
conda deactivate
echo '...done.'