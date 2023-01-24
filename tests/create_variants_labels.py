import os
import json
import numpy as np
import pandas as pd
import pyranges as pr
from operator import itemgetter


def create_variance_labels(data_dir, split_label, variants_file):
    with open(os.path.join(data_dir, 'statistics.json')) as params_open:
        params = json.load(params_open)

    variants_ranges = {'twi.24': [], 'ctcf.68': [], 'mef2.68': [], 'mef2.1012': [], 'bin.68': [], 'bin.1012': []}
    indices = [0, 1, 2, 8]
    skip = True
    with open(variants_file, 'r') as f:
        for line in f.readlines():
            if skip:
                skip = False
                continue
            line_divided = line.replace('"', '').strip().split(' ')
            condition = line_divided[5]
            line_divided = ' '.join(itemgetter(*indices)(line_divided))
            variants_ranges[condition].append(line_divided)

        # create ranges only for insignificant variants
    gr_nosig = {n: pr.from_string('\n'.join(['Chromosome Start End'] + [x[:-6] for x in s if x.split()[-1] == 'FALSE']))
                for n, s in variants_ranges.items()}
    # create ranges only for significant variants
    gr_sig = {n: pr.from_string('\n'.join(['Chromosome Start End'] + [x[:-5] for x in s if x.split()[-1] == 'TRUE']))
              for n, s in variants_ranges.items()}

    chr = []
    starts = []
    ends = []
    num_sequences = 0
    with open(os.path.join(data_dir, 'sequences.bed'), 'r') as f:
        for line in f.readlines():
            parts = line.split()
            if parts[-1] != split_label:
                continue
            num_sequences += 1
            for i in range(int(parts[1]), int(parts[2]), 128):
                chr.append(parts[0])
                starts.append(str(i))
                ends.append(str(i + 128))
    gr = pr.PyRanges(chromosomes=chr, starts=starts, ends=ends)

    overlaps_nosig = pr.count_overlaps(gr_nosig, gr).df
    overlaps_sig = pr.count_overlaps(gr_sig, gr).df

    df = pd.DataFrame(list(zip(chr, starts, ends)), columns=['Chromosome', 'Start', 'End'])
    df = df.astype({"Chromosome": str, "Start": int, "End": int})
    nosig_df = pd.merge(df, overlaps_nosig, how='left', left_on=['Chromosome', 'Start', 'End'],
                        right_on=['Chromosome', 'Start', 'End'])
    sig_df = pd.merge(df, overlaps_sig, how='left', left_on=['Chromosome', 'Start', 'End'],
                      right_on=['Chromosome', 'Start', 'End'])

    labels = {'twi.24', 'ctcf.68', 'mef2.68', 'mef2.1012', 'bin.68', 'bin.1012'}

    for l in labels:
        tag_labels = [2 if sig_x > 0 else 1 if nosig_x > 0 else 0 for sig_x, nosig_x in
                      zip(sig_df[l].tolist(), nosig_df[l].tolist())]
        array = np.array(tag_labels)
        array = array.reshape((num_sequences, params['target_length']))
        os.makedirs(os.path.join(data_dir, 'labels'), exist_ok=True)
        np.savez_compressed(os.path.join(data_dir, 'labels', f'{l}_variants.npz'), data=array)


if __name__ == '__main__':
    # data_dir = '/g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila_l131k'
    data_dir = '/g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l65k'
    # data_dir = '/g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila_l32k'
    variants_file = '/g/scb/zaugg/stojanov/basenji/experiments/data/quantified_variants_all_conditions.txt'
    split_label = 'test'
    create_variance_labels(data_dir, split_label, variants_file)
