import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve


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


def scatter_lims(vals1, vals2=None, buffer=.05):
    if vals2 is not None:
        vals = np.concatenate((vals1, vals2))
    else:
        vals = vals1
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    buf = .05 * (vmax - vmin)

    if vmin == 0:
        vmin -= buf / 2
    else:
        vmin -= buf
    vmax += buf

    return vmin, vmax


def regplot(vals1, vals2, out_pdf, poly_order=1, alpha=0.5, point_size=10, colors=None,
            cor='pearsonr', print_sig=False, square=False, x_label=None, y_label=None, title=None,
            figsize=(6, 6), sample=None, table=False, tight=False):
    if table:
        out_txt = '%s.txt' % out_pdf[:-4]
        out_open = open(out_txt, 'w')
        for i in range(len(vals1)):
            print(vals1[i], vals2[i], file=out_open)
        out_open.close()

    if sample is not None and sample < len(vals1):
        indexes = np.random.choice(np.arange(0, len(vals1)), sample, replace=False)
        vals1 = vals1[indexes]
        vals2 = vals2[indexes]

    plt.figure(figsize=figsize)

    gold = sns.color_palette('husl', 8)[1]

    if colors is None:
        ax = sns.regplot(vals1, vals2, color='black',
                         order=poly_order,
                         scatter_kws={'color': 'black',
                                      's': point_size,
                                      'alpha': alpha},
                         line_kws={'color': gold})
    else:
        plt.scatter(vals1, vals2, c=colors,
                    s=point_size, alpha=alpha, cmap='RdBu')
        plt.colorbar()
        ax = sns.regplot(vals1, vals2,
                         scatter=False, order=poly_order,
                         line_kws={'color': gold})

    if square:
        xmin, xmax = scatter_lims(vals1, vals2)
        ymin, ymax = xmin, xmax
    else:
        xmin, xmax = scatter_lims(vals1)
        ymin, ymax = scatter_lims(vals2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        plt.title(title)

    if cor is None:
        corr = None
    elif cor.lower() in ['spearman', 'spearmanr']:
        corr, csig = spearmanr(vals1, vals2)
        corr_str = 'SpearmanR: %.3f' % corr
    elif cor.lower() in ['pearson', 'pearsonr']:
        corr, csig = pearsonr(vals1, vals2)
        corr_str = 'PearsonR: %.3f' % corr
    else:
        corr = None

    if print_sig:
        if csig > .001:
            corr_str += '\n p %.3f' % csig
        else:
            corr_str += '\n p %.1e' % csig

    if corr is not None:
        xlim_eps = (xmax - xmin) * .03
        ylim_eps = (ymax - ymin) * .05

        ax.text(
            xmin + xlim_eps,
            ymax - 2 * ylim_eps,
            corr_str,
            horizontalalignment='left',
            fontsize=12)

    # ax.grid(True, linestyle=':')
    sns.despine()

    if tight:
        plt.tight_layout()

    plt.savefig(out_pdf)
    plt.close()


def test_peaks(test_preds, test_targets_peaks, out_dir, suffix):
    os.makedirs(os.path.join(out_dir, 'violin_plots'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'roc_plots'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pr_plots'), exist_ok=True)
    # sample every few bins to decrease correlations
    ds_indexes = np.arange(0, test_preds.shape[1], 1)

    # subset and flatten
    test_preds_ti_flat = test_preds[:, ds_indexes].flatten().astype('float32')
    test_targets_peaks = test_targets_peaks[:, ds_indexes].flatten()
    test_targets_peaks_str = np.where(test_targets_peaks, 'Peak', 'Background')

    # violin plot
    sns.set(font_scale=1.3, style='ticks')
    plt.figure()
    df = pd.DataFrame({
        'log2 Prediction': np.log2(test_preds_ti_flat + 1),
        'Experimental coverage status': test_targets_peaks_str
    })
    ax = sns.violinplot(
        x='Experimental coverage status', y='log2 Prediction', data=df)
    ax.grid(True, linestyle=':')
    plt.savefig(os.path.join(out_dir, 'violin_plots', f'{suffix}.pdf'))
    plt.close()

    # ROC
    plt.figure()
    fpr, tpr, _ = roc_curve(test_targets_peaks, test_preds_ti_flat)
    auroc = roc_auc_score(test_targets_peaks, test_preds_ti_flat)
    plt.plot(
        [0, 1], [0, 1], c='black', linewidth=1, linestyle='--', alpha=0.7)
    plt.plot(fpr, tpr, c='black')
    ax = plt.gca()
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.text(
        0.99, 0.02, 'AUROC %.3f' % auroc,
        horizontalalignment='right')  # , fontsize=14)
    ax.grid(True, linestyle=':')
    plt.savefig(os.path.join(out_dir, 'roc_plots', f'{suffix}.pdf'))
    plt.close()

    # PR
    plt.figure()
    prec, recall, _ = precision_recall_curve(test_targets_peaks,
                                             test_preds_ti_flat)
    auprc = average_precision_score(test_targets_peaks, test_preds_ti_flat)
    plt.axhline(
        y=test_targets_peaks.mean(),
        c='black',
        linewidth=1,
        linestyle='--',
        alpha=0.7)
    plt.plot(recall, prec, c='black')
    ax = plt.gca()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.text(
        0.99, 0.95, 'AUPRC %.3f' % auprc,
        horizontalalignment='right')  # , fontsize=14)
    ax.grid(True, linestyle=':')
    plt.savefig(os.path.join(out_dir, 'pr_plots', f'{suffix}.pdf'))
    plt.close()

    if test_targets_peaks.sum() == 0:
        auroc = 0.5
        auprc = 0
    else:
        # compute prediction accuracy
        auroc = roc_auc_score(test_targets_peaks, test_preds_ti_flat)
        auprc = average_precision_score(test_targets_peaks, test_preds_ti_flat)

    print('%6d  %.5f  %.5f' % (test_targets_peaks.sum(), auroc, auprc))


def plot_scatter(test_preds, test_targets, out_dir, suffix):
    os.makedirs(os.path.join(out_dir, 'scatter_plots'), exist_ok=True)
    # sample every few bins to decrease correlations
    ds_indexes = np.arange(0, test_preds.shape[1], 1)

    # subset and flatten
    test_preds_flat = test_preds[:, ds_indexes].flatten().astype('float32')
    test_targets_flat = test_targets[:, ds_indexes].flatten().astype('float32')

    # take log2
    test_targets_log = np.log2(test_targets_flat + 1)
    test_preds_log = np.log2(test_preds_flat + 1)

    # plot
    sns.set(font_scale=1.2, style='ticks')
    out_pdf = os.path.join(out_dir, 'scatter_plots', f'{suffix}.pdf')
    regplot(
        test_targets_flat,
        test_preds_flat,
        out_pdf,
        poly_order=1,
        alpha=0.3,
        sample=500,
        figsize=(6, 6),
        x_label='Experiment',
        y_label='Prediction',
        table=True)

    # plot log2
    sns.set(font_scale=1.2, style='ticks')
    out_pdf = os.path.join(out_dir, 'scatter_plots', f'{suffix}_log.pdf')
    regplot(
        test_targets_log,
        test_preds_log,
        out_pdf,
        poly_order=1,
        alpha=0.3,
        sample=500,
        figsize=(6, 6),
        x_label='log2 Experiment',
        y_label='log2 Prediction',
        table=True)


if __name__ == '__main__':
    # data_dir = '/g/scb/zaugg/stojanov/basenji/experiments/data/drosophila_l131k/'
    data_dir = r'/experiments/data/drosophila_l131k'
    # model_dir = '/g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_augmented'
    model_dir = r'/experiments/models/drosophila_l131k_augmented'
    f = h5py.File(os.path.join(model_dir, 'preds.h5'))
    test_preds = f['preds'][:]
    f = h5py.File(os.path.join(model_dir, 'targets.h5'))
    test_gt = f['targets'][:]
    automated_peaks = True

    mask = []
    with open(os.path.join(data_dir, 'sequences.bed'), 'r') as f:
        for line in f.readlines():
            parts = line.split()
            if parts[-1] != 'test':
                continue
            mask.append(0 if parts[0] in ['chrX', 'chrY'] else 1)
    mask = np.array(mask)

    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152}
                        #'CTCF.embryo:FBgn0035769': 69, 'twi.embryo_2h-4h-AEL_homemade:FBgn0003900': 907}
    for label, ind in important_tracks.items():
        test_gt_ti = test_gt[:, :, ind]
        test_preds_i = test_preds[:, :, ind]
        plot_scatter(test_preds_i[np.where(mask == 1)], test_gt_ti[np.where(mask == 1)], model_dir, label)
        if automated_peaks:
            test_gt_ti_flat = test_gt_ti.flatten().astype('float32')
            # call peaks
            test_targets_ti_lambda = np.mean(test_gt_ti_flat)
            test_targets_pvals = 1 - poisson.cdf(
                np.round(test_gt_ti_flat) - 1, mu=test_targets_ti_lambda)
            test_targets_qvals = np.array(ben_hoch(test_targets_pvals))
            test_targets_peaks = test_targets_qvals < 0.01
            test_targets_peaks = test_targets_peaks.reshape(test_gt_ti.shape)
            test_targets_peaks = test_targets_peaks[np.where(mask == 1)]
        else:
            test_targets_peaks = np.load(os.path.join(data_dir, 'labels', f'{label}_peaks.npz'))['data']
            test_targets_peaks = test_targets_peaks[np.where(mask == 1)]
            test_targets_peaks = test_targets_peaks > 0

        test_preds_i = test_preds_i[np.where(mask == 1)]
        test_peaks(test_preds_i, test_targets_peaks, model_dir, label + '_automated' if automated_peaks else label)
