import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress


def plot_tracks(tracks, interval, height=1.5, save_path='out.png'):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(0, 896, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def evaluate(targets, test_targets, test_predictions, output_dir):
    # evaluate
    pearson = []
    r2 = []
    targets_names = list(targets.keys())
    for i in range(test_targets.shape[2]):
        pear_corr, sig = pearsonr(np.log2(test_targets[:, :, i].flatten() + 1),
                                  np.log2(test_predictions[:, :, i].flatten() + 1))
        _, _, r_value, _, _ = linregress(np.log2(test_targets[:, :, i].flatten() + 1),
                                         np.log2(test_predictions[:, :, i].flatten() + 1))
        pearson.append(pear_corr)
        r2.append(r_value ** 2)
        tracks = {f'F1 target - seq {0}': test_targets[0, :, i],
                  f'F1 pred - seq {0}': test_predictions[0, :, i]}
        plot_tracks(tracks, 'int', save_path=os.path.join(output_dir, f'tracks_{targets_names[i]}.png'))

    pear_corr, sig = pearsonr(np.log2(test_targets.flatten() + 1),
                              np.log2(test_predictions.flatten() + 1))
    _, _, r_value, _, _ = linregress(np.log2(test_targets.flatten() + 1),
                                     np.log2(test_predictions.flatten() + 1))
    print('Test pearson: ', pear_corr)
    print('Test r2: ', r_value ** 2)

    # write target-level statistics
    targets_acc_df = pd.DataFrame({
        'index': list(targets.values()),
        'pearsonr': pearson,
        'r2': r2,
        'identifier': list(targets.keys())
    })

    targets_acc_df.to_csv('%s/acc_log.txt' % output_dir, sep='\t',
                          index=False, float_format='%.5f')


def plot_results(dir_path, save_path):
    results = pd.read_csv('%s/acc_log.txt' % dir_path, sep='\t', index_col=0)

    sns.set(style="whitegrid")
    sns.set_palette("Paired")
    ax = sns.boxplot(data=results, y="pearsonr", x="identifier", showfliers=False)
    ax = sns.swarmplot(data=results, y="pearsonr", x="identifier", color=".25", alpha=0.5)
    plt.savefig(save_path)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    prediction_dir = '/g/furlong/project/103_Basenji/analysis/personalised_genomes_32k/'
    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152}
    for dir_ind in ['DGRP-28', 'DGRP-307', 'DGRP-399', 'DGRP-57', 'DGRP-639', 'DGRP-712', 'DGRP-714', 'DGRP-852']:
        f = h5py.File(os.path.join(prediction_dir, dir_ind, 'predict.h5'), 'r')
        test_preds = f['preds'][:]
        test_preds = test_preds[:, :, list(important_tracks.values())]
        gt_folder = rf'/g/furlong/project/103_Basenji/data/personalised_predictions_32k/{dir_ind}/'
        test_gt = []
        for tf in important_tracks.keys():
            f = h5py.File(os.path.join(gt_folder, f'{tf}_targets.h5'), 'r')
            test_gt.append(f['targets'][:])
            # Check what to do here!!!
        test_gt = np.nan_to_num(np.array(test_gt).astype('float16'))
        test_gt = np.rollaxis(test_gt, 0, test_gt.ndim)
        evaluate(important_tracks, test_gt, test_preds, os.path.join(prediction_dir, dir_ind))
        plot_results(os.path.join(prediction_dir, dir_ind),
                     os.path.join(os.path.join(prediction_dir, dir_ind), 'acc_log.png'))
