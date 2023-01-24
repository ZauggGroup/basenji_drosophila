import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from basenji import plots
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, poisson


def ben_hoch(p_values):
    """ Convert the given p-values to q-values using Benjamini-Hochberg FDR. """
    m = len(p_values)

    # attach original indexes to p-values
    p_k = [(p_values[k], k) for k in range(m)]

    # sort by p-value
    p_k.sort()

    # compute q-value and attach original index to front
    k_q = [(p_k[i][1], p_k[i][0] * m // (i + 1)) for i in range(m)]

    # re-sort by original index
    k_q.sort()

    # drop original indexes
    q_values = [k_q[k][1] for k in range(m)]

    return q_values


if __name__ == '__main__':
    reference_pred_dir = r'Y:\stojanov\basenji\experiments\personalized_predictions\reference\preds'
    personalised_pred_dir = r'Y:\stojanov\basenji\experiments\personalized_predictions\vgn\preds'
    f = h5py.File(os.path.join(reference_pred_dir, 'predict.h5'))
    reference_preds = f['preds'][:]
    f = h5py.File(os.path.join(personalised_pred_dir, 'predict.h5'))
    personalized_preds = f['preds'][:]

    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152}

    for track, ind in important_tracks.items():
        reference_current = reference_preds[:, :, ind]
        personalized_current = personalized_preds[:, :, ind]

        ############################################
        # scatter

        # sample every few bins (adjust to plot the # points I want)
        ds_indexes = np.arange(0, personalized_current.shape[1], 8)

        # subset and flatten
        reference_current_flat = reference_current[:, ds_indexes].flatten().astype('float32')
        personalized_current_flat = personalized_current[:, ds_indexes].flatten().astype('float32')

        # take log2
        ref_log = np.log2(reference_current_flat + 1)
        per_log = np.log2(personalized_current_flat + 1)
        # corr, sig = pearsonr(ref_log, per_log)

        # plot log2
        sns.set(font_scale=1.2, style='ticks')
        out_pdf = f'ref_vgn_{track}.png'
        plots.regplot(
            ref_log,
            per_log,
            out_pdf,
            poly_order=1,
            alpha=0.3,
            sample=500,
            figsize=(6, 6),
            x_label='log2 Experiment',
            y_label='log2 Prediction',
            table=True)
        ############################################
        # violin

        # call peaks
        ref_flat = reference_current.flatten()
        per_flat = personalized_current.flatten()
        ref_lambda = np.mean(ref_flat)
        ref_pvals = 1 - poisson.cdf(
            np.round(ref_flat) - 1, mu=ref_lambda)
        ref_qvals = np.array(ben_hoch(ref_pvals))
        ref_peaks = ref_qvals < 0.01
        ref_peaks_str = np.where(ref_peaks, 'Peak', 'Background')

        # violin plot
        sns.set(font_scale=1.3, style='ticks')
        plt.figure()
        df = pd.DataFrame({
            'log2 Prediction': np.log2(per_flat + 1),
            'Experimental coverage status': ref_peaks_str
        })
        ax = sns.violinplot(x='Experimental coverage status', y='log2 Prediction', data=df)
        ax.grid(True, linestyle=':')
        plt.savefig(f'ref_vgn_violin_{track}.png')
        plt.close()
