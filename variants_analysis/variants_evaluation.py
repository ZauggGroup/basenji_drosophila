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


def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(0, 896, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    return plt


def create_peaks_plots(reference_current, alternative_current, track):
    ############################################
    # scatter

    # sample every few bins (adjust to plot the # points I want)
    ds_indexes = np.arange(0, alternative_current.shape[1], 8)

    # subset and flatten
    reference_current_flat = reference_current[:, ds_indexes].flatten().astype('float32')
    alternative_current_flat = alternative_current[:, ds_indexes].flatten().astype('float32')

    # take log2
    ref_log = np.log2(reference_current_flat + 1)
    per_log = np.log2(alternative_current_flat + 1)
    # corr, sig = pearsonr(ref_log, per_log)

    # plot log2
    sns.set(font_scale=1.2, style='ticks')
    out_pdf = f'ref_alt_{track}.png'
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
    per_flat = alternative_current.flatten()
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
    plt.savefig(f'ref_alt_violin_{track}.png')
    plt.close()


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

    tracks = [('twi.24', important_tracks['twi.24']),
              ('bin.1012', important_tracks['bin.1012'])]
    variants_file = '/g/scb/zaugg/stojanov/basenji/experiments/data/quantified_variants_all_conditions.txt'
    variants = pd.read_csv(variants_file, sep=r'\s+', header=0, index_col=False)
    variants['variant_id'] = variants['variant_id'].str.replace('_', ':')

    for i, target in enumerate(target_ids):
        current = variants.loc[(variants['is_indel'] == False) & (variants['condition'] == target)]
        current_sads = sad_values[:, i]
        df = pd.DataFrame({'sad': current_sads, 'snp': snps})
        res = current.join(df.set_index('snp'), on='variant_id')
        res['label'] = ['AI>0.6' if ai_val > 0.6 else 'AI<0.4' if ai_val < 0.4 else '0.6>AI>0.4'
                        for ai_val in res['AI']]
        res['class'] = ['red' if ai_val > 0.6 else 'blue' if ai_val < 0.4 else 'black'
                        for ai_val in res['AI']]
        res = res[res.sad.notnull()]

        sads = res[['sad']].values.flatten()
        ais = res[['AI']].values.flatten()
        labels = res[['label']].values.flatten()
        classes = res[['class']].values.flatten()
        significant = res[['significant']].values.flatten()
        # plot
        # sns.set(font_scale=1.2, style='ticks')
        # out_pdf = f'sad_ai_{target}.png'
        # plots.regplot(
        #     sads,
        #     ais,
        #     out_pdf,
        #     poly_order=1,
        #     alpha=0.3,
        #     sample=None,
        #     figsize=(6, 6),
        #     colors=classes,
        #     x_label='Basenji prediction - SAD',
        #     y_label='AI score',
        #     table=True)
        df = pd.DataFrame(list(zip(sads, significant)), columns=['val', 'class'])
        sns.boxplot(data=df, x='class', y='val')
        plt.savefig(f'ref_alt_{target}.png')
        plt.close()

    # for reference_pred_dir, alternative_pred_dir, track, alt_bed_file, ref_bed_file in zip(reference_pred_dirs,
    #                                                                                        alternative_pred_dirs,
    #                                                                                        tracks, alt_bed_files,
    #                                                                                        ref_bed_files):
    #     ref_seqs = []
    #     with open(ref_bed_file, 'r') as f:
    #         for line in f.readlines():
    #             ref_seqs.append(line.split()[0].split(';')[1])
    #
    #     alt_seqs = {}
    #     with open(alt_bed_file, 'r') as f:
    #         for i, line in enumerate(f.readlines()):
    #             alt_seqs[line.split()[0].split(';')[1]] = i
    #
    #     f = h5py.File(os.path.join(reference_pred_dir, 'predict.h5'))
    #     reference_preds = f['preds'][:]
    #     f = h5py.File(os.path.join(alternative_pred_dir, 'predict.h5'))
    #     alt_preds = f['preds'][:]
    #
    #     track_label, track_ind = track
    #     ref_ti = reference_preds[:, :, track_ind]
    #     alt_ti = alt_preds[:, :, track_ind]
    #     # create_peaks_plots(ref_ti, alt_ti, track_label)
    #     current_peaks = peaks.loc[peaks['condition'] == track_label]
    #     diffs = []
    #     ais = []
    #     labels = []
    #     classes = []
    #     for i, seq in enumerate(ref_seqs):
    #         # tracks = {f'F1 REF - seq {i}': ref_ti[i, :],
    #         #           f'F1 ALT - seq {i}': alt_ti[i, :],
    #         #           f'F1 REF-ALT - seq {i}': ref_ti[i, :] - alt_ti[i, :]}
    #         # print(f'Seq {i}')
    #         # print('max', np.max(np.abs(ref_ti[i, :] - alt_ti[i, :])))
    #         # print('sum', np.sum(np.abs(ref_ti[i, :] - alt_ti[i, :])))
    #         # print('center', np.abs(ref_ti[i, 448] - alt_ti[i, 448]))
    #         # plt = plot_tracks(tracks, 'target_interval')
    #         # plt.savefig(f'track_seq{i}')
    #         current = variants.loc[
    #             (variants['variant_id'] == '_'.join(seq.split('_')[:2])) & (variants['condition'] == track_label)]
    #         if current.shape[0] == 0:
    #             continue
    #         for j in range(current.shape[0]):
    #             ai_val = current[['AI']].values[j, 0]
    #             significant = current[['significant']].values[j, 0]
    #             p_id = current[['peak_id']].values[j, 0]
    #             peak = current_peaks.loc[current_peaks['peak_id'] == p_id]
    #             start = 488 + math.ceil((current['start'].values[j] - peak['start'].values[0]) / 128)
    #             end = start + math.ceil((peak['end'].values[0] - peak['start'].values[0]) / 128) + 1
    #
    #             ref_val = ref_ti[i, start:end]
    #             alt_val = alt_ti[alt_seqs[seq], start:end]
    #             # diff = np.average(ref_val / (ref_val + alt_val + 1e-10))
    #             diff = np.average(ref_val - alt_val)
    #             diffs.append(diff)
    #             ais.append(ai_val)
    #             labels.append('blue' if significant else 'black')
    #             classes.append('AI>0.6' if ai_val > 0.6 else 'AI<0.4' if ai_val < 0.4 else '0.6>AI>0.4')
    #
    #     # diffs = np.nan_to_num(np.array(diffs))
    #     # ais = np.nan_to_num(np.array(ais))
    #     # # plot
    #     # sns.set(font_scale=1.2, style='ticks')
    #     # out_pdf = f'ref_alt_{track[0]}.png'
    #     # plots.regplot(
    #     #     diffs,
    #     #     ais,
    #     #     out_pdf,
    #     #     poly_order=1,
    #     #     alpha=0.3,
    #     #     sample=None,
    #     #     figsize=(6, 6),
    #     #     colors=labels,
    #     #     x_label='Basenji prediction - REF / (REF + ALT)',
    #     #     y_label='AI score',
    #     #     table=True)
    #     df = pd.DataFrame(list(zip(diffs, classes)), columns=['val', 'class'])
    #     print(df.head())
    #     sns.boxplot(data=df, x='class', y='val')
    #     plt.savefig(f'ref_alt_{track[0]}.png')
