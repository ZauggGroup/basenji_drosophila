import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(0, 896, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    return plt


if __name__ == '__main__':
    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152}
    sad_file = '/g/scb/zaugg/stojanov/basenji_drosophila/outputs/variants_predictions/sad_scores5/sad.h5'
    sad_values_file = h5py.File(sad_file, 'r')
    sad_values = sad_values_file['SAD'][:]
    snps = sad_values_file['snp'][:]
    snps = [x.decode("utf-8") for x in snps]
    target_ids = sad_values_file['target_ids'][:]
    target_ids = [x.decode("utf-8") for x in target_ids]
    sad_values_file.close()

    preds_file = '/g/scb/zaugg/stojanov/basenji_drosophila/outputs/variants_predictions/sad_scores5/preds.h5'
    preds_values_file = h5py.File(preds_file, 'r')
    preds_ref = preds_values_file['REF'][:]
    preds_alt = preds_values_file['ALT'][:]
    preds_values_file.close()

    variants_file = '/g/scb/zaugg/stojanov/basenji/experiments/data/quantified_variants_all_conditions.txt'
    variants = pd.read_csv(variants_file, sep=r'\s+', header=0, index_col=False)
    variants['variant_id'] = variants['variant_id'].str.replace('_', ':')

    print(target_ids)
    for i, target in enumerate(target_ids):
        if i != 2:
            continue
        current = variants.loc[(variants['is_indel'] == False) & (variants['condition'] == target)]
        current_sads = sad_values[:, i]
        df = pd.DataFrame({'sad': current_sads, 'snp': snps, 'index': list(range(current_sads.shape[0]))})
        res = current.join(df.set_index('snp'), on='variant_id')
        res = res[res.sad.notnull()]
        # res = res[(res['AI'] > 0.6) & (res['AI'] < 0.4)]
        res = res[res['AI'] > 0.6]
        res = res[res['sad'] < 100]
        indices = res['index'].values.flatten()
        print(indices)
        for ind in indices:
            ind = int(ind)
            plt = plot_tracks({'REF': preds_ref[ind, :, i], 'ALT': preds_alt[ind, :, i]}, 'target_interval')
            plt.savefig(f'track_{ind}_bigger.png')
