#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

from optparse import OptionParser
import json
import pdb
import pickle
import os
from queue import Queue
import sys
from threading import Thread
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

if tf.__version__[0] == '1':
    tf.compat.v1.enable_eager_execution()

from basenji import seqnn
from basenji import stream
from basenji import vcf as bvcf

'''
basenji_sad.py

Compute SNP Activity Difference (SAD) scores for SNPs in a VCF file.
'''


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='genome_fasta',
                      default='data/hg38.fa',
                      help='Genome FASTA for sequences [Default: %default]')
    parser.add_option('-n', dest='norm_file',
                      default=None,
                      help='Normalize SAD scores')
    parser.add_option('-o', dest='out_dir',
                      default='sad',
                      help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-p', dest='processes',
                      default=None, type='int',
                      help='Number of processes, passed by multi script')
    parser.add_option('--pseudo', dest='log_pseudo',
                      default=1, type='float',
                      help='Log2 pseudocount [Default: %default]')
    parser.add_option('--rc', dest='rc',
                      default=False, action='store_true',
                      help='Average forward and reverse complement predictions [Default: %default]')
    parser.add_option('--shifts', dest='shifts',
                      default='0', type='str',
                      help='Ensemble prediction shifts [Default: %default]')
    parser.add_option('--stats', dest='sad_stats',
                      default='SAD',
                      help='Comma-separated list of stats to save. [Default: %default]')
    parser.add_option('-t', dest='targets_file',
                      default=None, type='str',
                      help='File specifying target indexes and labels in table format')
    parser.add_option('--ti', dest='track_indexes',
                      default=None, type='str',
                      help='Comma-separated list of target indexes to output BigWig tracks')
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        vcf_file = args[2]
    else:
        parser.error('Must provide parameters and model files and QTL VCF file')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if options.track_indexes is None:
        options.track_indexes = []
    else:
        options.track_indexes = [int(ti) for ti in options.track_indexes.split(',')]
        if not os.path.isdir('%s/tracks' % options.out_dir):
            os.mkdir('%s/tracks' % options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(',')]
    options.sad_stats = options.sad_stats.split(',')

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']

    if options.targets_file is None:
        target_slice = None
    else:
        targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
        target_ids = targets_df.identifier
        target_labels = targets_df.description
        target_slice = targets_df.index

    #################################################################
    # setup model

    # can we sum on GPU?
    length_stats = set(['SAX', 'SAXR', 'SAR', 'ALT', 'REF'])
    sum_length = length_stats.isdisjoint(set(options.sad_stats))
    sum_length = False  # minimal influence

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    seqnn_model.build_slice(target_slice)
    if sum_length:
        seqnn_model.build_sad()
    seqnn_model.build_ensemble(options.rc, options.shifts)

    targets_length = seqnn_model.target_lengths[0]
    num_targets = seqnn_model.num_targets()
    if options.targets_file is None:
        target_ids = ['t%d' % ti for ti in range(num_targets)]
        target_labels = [''] * len(target_ids)

    #################################################################
    # load SNPs

    # filter for worker SNPs
    if options.processes is not None:
        # determine boundaries
        num_snps = bvcf.vcf_count(vcf_file)
        worker_bounds = np.linspace(0, num_snps, options.processes + 1, dtype='int')

        # read SNPs form VCF
        snps = bvcf.vcf_snps(vcf_file, start_i=worker_bounds[worker_index],
                             end_i=worker_bounds[worker_index + 1])

    else:
        # read SNPs form VCF
        snps = bvcf.vcf_snps(vcf_file)

    num_snps = len(snps)
    if os.path.exists('%s/preds.h5' % options.out_dir):
        preds_out = h5py.File('%s/preds.h5' % options.out_dir, 'r')
        ref_preds = preds_out['REF'][:]
        alt_preds = preds_out['ALT'][:]
    else:
        # open genome FASTA
        genome_open = pysam.Fastafile(options.genome_fasta)

        def snp_gen():
            for snp in snps:
                # get SNP sequences
                snp_1hot_list = bvcf.snp_seq1(snp, params_model['seq_length'], genome_open)
                for snp_1hot in snp_1hot_list:
                    yield snp_1hot

        #################################################################
        # predict SNP scores, write output

        # initialize predictions stream
        preds_stream = stream.PredStreamGen(seqnn_model, snp_gen(), params_train['batch_size'])

        # predictions index
        pi = 0
        ref_preds = []
        alt_preds = []

        for si in range(num_snps):
            # get predictions
            ref_preds.append(np.clip(preds_stream[pi], 0, 65500))
            pi += 1
            alt_preds.append(np.clip(preds_stream[pi], 0, 65500))
            pi += 1

        ref_preds = np.array(ref_preds).astype('float16')
        alt_preds = np.array(alt_preds).astype('float16')

        preds_out = h5py.File('%s/preds.h5' % options.out_dir, 'w')
        preds_out.create_dataset('REF', data=ref_preds, chunks=True)
        preds_out.create_dataset('ALT', data=alt_preds, chunks=True)
        preds_out.close()
        # close genome
        genome_open.close()

    # process SNP
    sad_vals = {sad_stat: [] for sad_stat in options.sad_stats}
    if sum_length:
        print('Length summed')
        out = calc_snp(ref_preds, alt_preds, options.sad_stats, options.log_pseudo)
        for k, o in out.items():
            sad_vals[k] = o
    else:
        out = calc_snp_len(ref_preds, alt_preds, options.sad_stats, options.log_pseudo)
        for k, o in out.items():
            sad_vals[k] = o

    #################################################################
    # setup output

    sad_out = initialize_output_h5(options.out_dir, options.sad_stats,
                                   snps, target_ids, target_labels, targets_length)

    ###################################################
    # compute SAD distributions across variants
    # initialize SAD stats
    for sad_stat in options.sad_stats:
        if sad_stat in ['REF', 'ALT']:
            sad_out.create_dataset(sad_stat, data=np.array(sad_vals[sad_stat], dtype='float16'))
        else:
            sad_out.create_dataset(sad_stat, data=np.array(sad_vals[sad_stat], dtype='float16'))

    write_pct(sad_out, options.sad_stats)
    sad_out.close()


def initialize_output_h5(out_dir, sad_stats, snps, target_ids, target_labels, targets_length):
    """Initialize an output HDF5 file for SAD stats."""

    num_targets = len(target_ids)
    num_snps = len(snps)

    sad_out = h5py.File('%s/sad.h5' % out_dir, 'w')

    # write SNPs
    snp_ids = np.array([snp.rsid for snp in snps], 'S')
    sad_out.create_dataset('snp', data=snp_ids)

    # write SNP chr
    snp_chr = np.array([snp.chr for snp in snps], 'S')
    sad_out.create_dataset('chr', data=snp_chr)

    # write SNP pos
    snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
    sad_out.create_dataset('pos', data=snp_pos)

    # check flips
    snp_flips = [snp.flipped for snp in snps]

    # write SNP reference allele
    snp_refs = []
    snp_alts = []
    for snp in snps:
        if snp.flipped:
            snp_refs.append(snp.alt_alleles[0])
            snp_alts.append(snp.ref_allele)
        else:
            snp_refs.append(snp.ref_allele)
            snp_alts.append(snp.alt_alleles[0])
    snp_refs = np.array(snp_refs, 'S')
    snp_alts = np.array(snp_alts, 'S')
    sad_out.create_dataset('ref_allele', data=snp_refs)
    sad_out.create_dataset('alt_allele', data=snp_alts)

    # write targets
    sad_out.create_dataset('target_ids', data=np.array(target_ids, 'S'))
    sad_out.create_dataset('target_labels', data=np.array(target_labels, 'S'))

    return sad_out


def write_pct(sad_out, sad_stats):
    """Compute percentile values for each target and write to HDF5."""

    # define percentiles
    d_fine = 0.001
    d_coarse = 0.01
    percentiles_neg = np.arange(d_fine, 0.1, d_fine)
    percentiles_base = np.arange(0.1, 0.9, d_coarse)
    percentiles_pos = np.arange(0.9, 1, d_fine)

    percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
    sad_out.create_dataset('percentiles', data=percentiles)
    pct_len = len(percentiles)

    for sad_stat in sad_stats:
        if sad_stat not in ['REF', 'ALT']:
            sad_stat_pct = '%s_pct' % sad_stat

            # compute
            sad_pct = np.percentile(sad_out[sad_stat], 100 * percentiles, axis=0).T
            sad_pct = sad_pct.astype('float16')

            # save
            sad_out.create_dataset(sad_stat_pct, data=sad_pct, dtype='float16')


def calc_snp(ref_preds_sum, alt_preds_sum, sad_stats, log_pseudo):
    """Write SNP predictions to HDF, assuming the length dimension has
        been collapsed."""
    sad_vals = {}
    # compare reference to alternative via mean subtraction
    if 'SAD' in sad_stats:
        sad = alt_preds_sum - ref_preds_sum
        sad_vals['SAD'] = sad.astype('float16')

    # compare reference to alternative via mean log division
    if 'SADR' in sad_stats:
        sar = np.log2(alt_preds_sum + log_pseudo) \
              - np.log2(ref_preds_sum + log_pseudo)
        sad_vals['SADR'] = sar.astype('float16')
    return sad_vals


def calc_snp_len(ref_preds, alt_preds, sad_stats, log_pseudo):
    """Calculate SNP predictions, assuming the length dimension has
        been maintained."""
    sad_vals = {}
    ref_preds = ref_preds.astype('float64')
    alt_preds = alt_preds.astype('float64')
    num_targets = ref_preds.shape[-1]

    # sum across length
    ref_preds_sum = ref_preds.sum(axis=1)
    alt_preds_sum = alt_preds.sum(axis=1)

    # compare reference to alternative via mean subtraction
    if 'SAD' in sad_stats:
        sad = alt_preds_sum - ref_preds_sum
        sad_vals['SAD'] = sad.astype('float16')

    # compare reference to alternative via max subtraction
    if 'SAX' in sad_stats:
        sad_vec = (alt_preds - ref_preds)
        max_i = np.argmax(np.abs(sad_vec), axis=1)
        sax = sad_vec[max_i, np.arange(num_targets)]
        sad_vals['SAX'] = sax.astype('float16')

    # compare reference to alternative via mean log division
    if 'SADR' in sad_stats:
        sar = np.log2(alt_preds_sum + log_pseudo) \
              - np.log2(ref_preds_sum + log_pseudo)
        sad_vals['SADR'] = sar.astype('float16')

    # compare reference to alternative via max subtraction
    if 'SAXR' in sad_stats:
        sar_vec = np.log2(alt_preds + log_pseudo) \
                  - np.log2(ref_preds + log_pseudo)
        max_i = np.argmax(np.abs(sar_vec), axis=1)
        saxr = sar_vec[max_i, np.arange(num_targets)]
        sad_vals['SAXR'] = saxr.astype('float16')

    # compare geometric means
    if 'SAR' in sad_stats:
        sar_vec = np.log2(alt_preds + log_pseudo) \
                  - np.log2(ref_preds + log_pseudo)
        geo_sad = sar_vec.sum(axis=1)
        sad_vals['SAR'] = geo_sad.astype('float16')

    # predictions
    if 'REF' in sad_stats:
        sad_vals['REF'] = ref_preds.astype('float16')
    if 'ALT' in sad_stats:
        sad_vals['ALT'] = alt_preds.astype('float16')
    return sad_vals


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
