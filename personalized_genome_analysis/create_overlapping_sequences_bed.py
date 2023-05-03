import os
import h5py
import pyBigWig
import numpy as np
from Bio import SeqIO


def write_seqs_bed(bed_file, seqs, labels=False):
    """Write sequences to BED file
    :param bed_file: bed file path to save the results
    :type bed_file: str
    :param seqs: sequence information for the bed file, including keys 'chr', 'start', 'end' and (optional 'label')
    :type seqs: list(dict)
    :param labels: if there are label to be saved in the bed file
    :type labels: bool
    :return: None
    :rtype: None
    """
    bed_out = open(bed_file, 'w')
    for i in range(len(seqs)):
        line = '%s\t%d\t%d' % (seqs[i]['chr'], seqs[i]['start'], seqs[i]['end'])
        if labels:
            line += '\t%s' % seqs[i]['label']
        print(line, file=bed_out)
    bed_out.close()


def create_and_save_seqs(genome_file, needed_chromosomes, seq_len, seqs_bed_file):
    """ Create a bed file with genome coordinates
    :param genome_file: path to the genome fasta file
    :type genome_file: str
    :param needed_chromosomes: list of chromosomes that need to be included
    :type needed_chromosomes: list(str)
    :param seq_len:
    :type seq_len: int
    :param seqs_bed_file: bed file path to save the sequence ranges
    :type seqs_bed_file: str
    :return: None
    :rtype: None
    """
    seqs = []
    for record in SeqIO.parse(genome_file, 'fasta'):
        if needed_chromosomes is not None and record.id not in needed_chromosomes:
            continue
        chr_len = len(record.seq)
        print('Len', chr_len)
        for i in range(0, chr_len - seq_len - 1, seq_len):
            start_ind = i
            end_ind = start_ind + seq_len
            seqs.append({'chr': record.id, 'start': start_ind, 'end': end_ind})

    write_seqs_bed(seqs_bed_file, seqs, False)
    return seqs


def extract_ground_truth(seqs, file, seq_len, crop_bp, pool_width, save_file_path):
    """Create a ground truth coverage .h5 file for the given
    :param seqs:  sequence information for the bed file, including keys 'chr', 'start', 'end' and (optional 'label')
    :type seqs: list(dict)
    :param file: bigWig coverage file
    :type file: str
    :param seq_len: length of the genome sequence included in one data point
    :type seq_len: int
    :param crop_bp: cropping of the output to match the desired output size
    :type crop_bp: int
    :param pool_width: pool width value to sum the output values
    :type pool_width: int
    :param save_file_path: file path to save the ground truth coverage data
    :type save_file_path: str
    :return: None
    :rtype: None
    """
    seq_len -= 2 * crop_bp
    target_length = seq_len // pool_width
    targets = []
    bw = pyBigWig.open(file)
    for s in seqs:
        # print(int(s['start']), int(s['end']))
        vals = np.array(bw.values(s['chr'], int(s['start']), int(s['end'])))
        # crop the bp from the ends
        vals = vals[crop_bp:-crop_bp]
        vals = vals.reshape(target_length, pool_width)
        # sum pool width values
        vals = vals.sum(axis=1, dtype='float32')
        # clip the values to float16
        vals = np.clip(vals, np.finfo(np.float16).min, np.finfo(np.float16).max)
        targets.append(vals.astype('float16'))
    targets = np.array(targets, dtype='float16')
    extreme_clip = np.percentile(targets, 100 * 0.9999999)
    targets = np.clip(targets, -extreme_clip, extreme_clip)
    seqs_cov_open = h5py.File(save_file_path, 'w')
    seqs_cov_open.create_dataset('targets', data=targets, dtype='float16', compression='gzip')


def extract_with_ground_truth(genome_folder, bigwig_folder, needed_chromosomes, seq_len,
                              crop_bp, pool_width, bed_folder):
    """Extract the information from the given genome file and coverage bigwig files
    :param genome_folder: path to the folder that contains genome fasta files
    :type genome_folder: str
    :param bigwig_folder: path to the folder that contains the ground truth bigwig coverage files
    :type bigwig_folder: str
    :param needed_chromosomes: list of chromosomes that need to be included
    :type needed_chromosomes: list(str)
    :param seq_len: length of the genome sequence included in one data point
    :type seq_len: int
    :param crop_bp: cropping of the output to match the desired output size
    :type crop_bp: int
    :param pool_width: pool width value to sum the output values
    :type pool_width: int
    :param bed_folder: path to the folder to save the bed files
    :type bed_folder: str
    :return: None
    :rtype: None
    """
    for file in ['DGRP-28', 'DGRP-307', 'DGRP-399', 'DGRP-57', 'DGRP-639', 'DGRP-712', 'DGRP-714', 'DGRP-852', 'vgn']:
        genome_file = os.path.join(genome_folder, f'{file}.fa')
        out_bed_folder = os.path.join(bed_folder, file)
        os.makedirs(out_bed_folder, exist_ok=True)
        seqs_bed_file = os.path.join(out_bed_folder, 'sequences.bed')
        seqs = create_and_save_seqs(genome_file, needed_chromosomes, seq_len, seqs_bed_file)
        if file != 'vgn':
            for tf in ['bin.1012', 'bin.68', 'ctcf.68', 'mef2.1012', 'mef2.68', 'twi.24']:
                bw_file = os.path.join(bigwig_folder,
                                       f'{tf}.vgn_{file.split("-")[1]}_mean_rep.filtered.rmdup_withUMI_AS_{file}_liftOver.bw')
                extract_ground_truth(seqs, bw_file, seq_len, crop_bp, pool_width,
                                     os.path.join(out_bed_folder, f'{tf}_targets.h5'))
        else:
            for tf in ['bin.1012', 'bin.68', 'ctcf.68', 'mef2.1012', 'mef2.68', 'twi.24']:
                for num in ['28', '307', '399', '57', '639', '712', '714', '852']:
                    bw_file = os.path.join(bigwig_folder,
                                           f'{tf}.vgn_{num}_mean_rep.filtered.rmdup_withUMI_AS_vgn_liftOver.bw')
                    extract_ground_truth(seqs, bw_file, seq_len, crop_bp, pool_width,
                                         os.path.join(out_bed_folder, f'{tf}_targets_{num}.h5'))


def extract_without_ground_truth(genome_folder, needed_chromosomes, seq_len, bed_folder):
    for file in ['DGRP-28', 'DGRP-307', 'DGRP-399', 'DGRP-57', 'DGRP-639', 'DGRP-712', 'DGRP-714', 'DGRP-852', 'vgn']:
        genome_file = os.path.join(genome_folder, f'{file}.fa')
        out_bed_folder = os.path.join(bed_folder, file)
        os.makedirs(out_bed_folder, exist_ok=True)
        seqs_bed_file = os.path.join(out_bed_folder, 'sequences.bed')
        _ = create_and_save_seqs(genome_file, needed_chromosomes, seq_len, seqs_bed_file)


if __name__ == '__main__':
    genome_folder = '/g/furlong/project/103_Basenji/data/personalised_genomes/'
    bigwig_folder = '/g/furlong/project/103_Basenji/data/personalised_genomes_chipseq/merged'
    out_bed_folder = '/g/furlong/project/103_Basenji/data/personalised_predictions_32k/'
    needed_chromosomes = ['chr2R']
    seq_len = 32768
    crop_bp = 2048
    pool_width = 128
    extract_with_ground_truth(genome_folder, bigwig_folder, needed_chromosomes, seq_len,
                              crop_bp, pool_width, out_bed_folder)
