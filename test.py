import os
import h5py
import numpy as np
import pandas as pd
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


if __name__ == '__main__':
    # model_dir = '/g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_new_data_augmented'
    model_dir = r'Y:\stojanov\basenji\experiments\models\drosophila_l131k_new_data_augmented'
    # targets_file = '/g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila.txt'
    targets_file = r'Y:\stojanov\basenji\experiments\data\new_data\drosophila.txt'
    f = h5py.File(os.path.join(model_dir, 'preds.h5'), 'r')
    test_preds = f['preds'][:]
    f = h5py.File(os.path.join(model_dir, 'targets.h5'), 'r')
    test_gt = f['targets'][:]

    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')

    evaluate(targets_df, test_gt, test_preds, model_dir)
