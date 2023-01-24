import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress


def evaluate(targets_df, test_targets, test_predictions, output_dir):
    test_predictions = np.clip(test_predictions, 0, 65500)
    # evaluate
    pearson = []
    r2 = []
    for i in range(test_targets.shape[2]):
        pear_corr, sig = pearsonr(np.log2(test_targets[:, :, i].flatten() + 1),
                                  np.log2(test_predictions[:, :, i].flatten() + 1))
        _, _, r_value, _, _ = linregress(np.log2(test_targets[:, :, i].flatten() + 1),
                                         np.log2(test_predictions[:, :, i].flatten() + 1))
        pearson.append(pear_corr)
        r2.append(r_value ** 2)

    pear_corr, sig = pearsonr(np.log2(test_targets.flatten() + 1),
                              np.log2(test_predictions.flatten() + 1))
    _, _, r_value, _, _ = linregress(np.log2(test_targets.flatten() + 1),
                                     np.log2(test_predictions.flatten() + 1))
    print('Test pearson: ', pear_corr)
    print('Test r2: ', r_value ** 2)

    # write target-level statistics
    targets_acc_df = pd.DataFrame({
        'index': targets_df.index,
        'pearsonr': pearson,
        'r2': r2,
        'identifier': targets_df.identifier,
        'description': targets_df.description
    })

    targets_acc_df.to_csv('%s/acc_log.txt' % output_dir, sep='\t',
                          index=False, float_format='%.5f')


def plot_results(dir_path, save_path, important_tracks):
    results = pd.read_csv('%s/acc_log.txt' % dir_path, sep='\t', index_col=0)
    if important_tracks is not None:
        results = results.filter(items=important_tracks, axis=0)

    sns.set(style="whitegrid")
    sns.set_palette("Paired")
    ax = sns.boxplot(data=results, y="pearsonr", x="description", showfliers=False)
    ax = sns.swarmplot(data=results, y="pearsonr", x="description", color=".25", alpha=0.5)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # model_dir = '/g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented'
    model_dir = r'/experiments/models/drosophila_l32k_augmented'
    # targets_file = '/g/scb/zaugg/stojanov/basenji/experiments/data/drosophila.txt'
    targets_file = r'/experiments/data/drosophila.txt'
    f = h5py.File(os.path.join(model_dir, 'preds.h5'), 'r')
    test_preds = f['preds'][:]
    f = h5py.File(os.path.join(model_dir, 'targets.h5'), 'r')
    test_gt = f['targets'][:]

    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')

    # evaluate(targets_df, test_gt, test_preds, model_dir)
    important_tracks = [12, 94, 271, 467, 628, 1152]
    plot_results(model_dir, os.path.join(model_dir, 'acc_log.png'), None)
