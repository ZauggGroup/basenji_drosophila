import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr


def evaluate_variants(test_preds, test_gt, mask, data_dir):
    test_preds = test_preds[np.where(mask == 1)]
    test_gt = test_gt[np.where(mask == 1)]
    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152}
    no_var = []
    no_sig_var = []
    sig_var = []
    for track, ind in important_tracks.items():
        variants_labels = np.load(os.path.join(data_dir, f'{track}_variants.npz'))['data']
        variants_labels = variants_labels[np.where(mask == 1)]
        test_gt_current = test_gt[:, :, ind]
        test_pred_current = test_preds[:, :, ind]
        corr, sig = pearsonr(np.log2(test_gt_current[variants_labels == 0].flatten() + 1),
                             np.log2(test_pred_current[variants_labels == 0].flatten() + 1))
        no_var.append(corr)
        corr, sig = pearsonr(np.log2(test_gt_current[variants_labels == 1].flatten() + 1),
                             np.log2(test_pred_current[variants_labels == 1].flatten() + 1))
        no_sig_var.append(corr)
        corr, sig = pearsonr(np.log2(test_gt_current[variants_labels == 2].flatten() + 1),
                             np.log2(test_pred_current[variants_labels == 2].flatten() + 1))
        sig_var.append(corr)
        print(f'{track}, no variants: {no_var[-1]}, variants: {no_sig_var[-1]}, significant variants: {sig_var[-1]}')

    sns.set(style="whitegrid")

    data = no_var + no_sig_var + sig_var
    labels = ['no variants'] * len(no_var) + ['variants'] * len(no_sig_var) + ['significant variants'] * len(sig_var)
    df = pd.DataFrame(list(zip(data, labels)), columns=['PearsonR', 'Type'])

    ax = sns.boxplot(x="Type", y="PearsonR", data=df, showfliers=False)
    ax = sns.swarmplot(x="Type", y="PearsonR", data=df, color=".25")
    plt.show()


def evaluate_variants_all_lines(test_preds, test_gt, mask, data_dir):
    test_preds = test_preds[np.where(mask == 1)]
    test_gt = test_gt[np.where(mask == 1)]
    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152,
                        "bin.1012.vgn_57": 1230, "bin.68.399_399": 1231, "bin.1012.vgn_852": 1232,
                        "twi.24.vgn_57": 1233, "mef2.1012.vgn_712": 1234, "bin.1012.vgn_28": 1235,
                        "ctcf.68.vgn_399": 1236, "mef2.1012.vgn_639": 1237, "ctcf.68.vgn_307": 1238,
                        "ctcf.68.vgn_28": 1239, "mef2.1012.vgn_852": 1240, "bin.68.vgn_714": 1241,
                        "bin.1012.399_399": 1242, "bin.1012.vgn_712": 1243, "mef2.68.399_399": 1244,
                        "mef2.68.vgn_28": 1245, "ctcf.68.399_399": 1246, "mef2.68.vgn_712": 1247,
                        "mef2.1012.vgn_vgn": 1248, "mef2.1012.vgn_57": 1249, "twi.24.vgn_852": 1250,
                        "mef2.1012.vgn_399": 1251, "bin.68.vgn_399": 1252, "bin.68.vgn_28": 1253,
                        "twi.24.vgn_714": 1254, "twi.24.vgn_639": 1255, "mef2.1012.vgn_714": 1256,
                        "bin.1012.vgn_399": 1257, "mef2.1012.399_399": 1258, "bin.1012.vgn_714": 1259,
                        "ctcf.68.vgn_852": 1260, "bin.1012.vgn_639": 1261, "ctcf.68.vgn_57": 1262,
                        "twi.24.vgn_vgn": 1263, "ctcf.68.vgn_714": 1264, "bin.1012.vgn_307": 1265,
                        "mef2.68.vgn_399": 1266, "mef2.68.vgn_vgn": 1267, "ctcf.68.vgn_vgn": 1268,
                        "bin.68.vgn_712": 1269, "ctcf.68.vgn_712": 1270, "mef2.68.vgn_714": 1271,
                        "mef2.1012.vgn_307": 1272, "bin.68.vgn_57": 1273, "ctcf.68.vgn_639": 1274,
                        "mef2.1012.vgn_28": 1275, "mef2.68.vgn_852": 1276, "twi.24.vgn_28": 1277,
                        "mef2.68.vgn_307": 1278, "bin.68.vgn_vgn": 1279, "bin.68.vgn_852": 1280, "twi.24.vgn_712": 1281,
                        "mef2.68.vgn_639": 1282, "twi.24.399_399": 1283, "bin.68.vgn_639": 1284, "twi.24.vgn_307": 1285,
                        "bin.1012.vgn_vgn": 1286, "mef2.68.vgn_57": 1287, "bin.68.vgn_307": 1288,
                        "twi.24.vgn_399": 1289}
    no_var = []
    no_sig_var = []
    sig_var = []
    for track, ind in important_tracks.items():
        track_name = '.'.join(track.split('.')[:2])
        variants_labels = np.load(os.path.join(data_dir, f'{track_name}_variants.npz'))['data']
        variants_labels = variants_labels[np.where(mask == 1)]
        test_gt_current = test_gt[:, :, ind]
        test_pred_current = test_preds[:, :, ind]
        corr, sig = pearsonr(np.log2(test_gt_current[variants_labels == 0].flatten() + 1),
                             np.log2(test_pred_current[variants_labels == 0].flatten() + 1))
        no_var.append(corr)
        corr, sig = pearsonr(np.log2(test_gt_current[variants_labels == 1].flatten() + 1),
                             np.log2(test_pred_current[variants_labels == 1].flatten() + 1))
        no_sig_var.append(corr)
        corr, sig = pearsonr(np.log2(test_gt_current[variants_labels == 2].flatten() + 1),
                             np.log2(test_pred_current[variants_labels == 2].flatten() + 1))
        sig_var.append(corr)
        print(f'{track}, no variants: {no_var[-1]}, variants: {no_sig_var[-1]}, significant variants: {sig_var[-1]}')

    sns.set(style="whitegrid")

    data = no_var + no_sig_var + sig_var
    labels = ['no variants'] * len(no_var) + ['variants'] * len(no_sig_var) + ['significant variants'] * len(sig_var)
    df = pd.DataFrame(list(zip(data, labels)), columns=['PearsonR', 'Type'])

    ax = sns.boxplot(x="Type", y="PearsonR", data=df, showfliers=False)
    ax = sns.swarmplot(x="Type", y="PearsonR", data=df, color=".25")
    plt.show()


if __name__ == '__main__':
    # data_dir = '/g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila_l131k/labels'
    data_dir = r'Y:\stojanov\basenji\experiments\data\new_data\drosophila_l131k'
    # model_dir = '/g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_new_data_augmented'
    model_dir = r'Y:\stojanov\basenji\experiments\models\drosophila_l131k_new_data_augmented'
    f = h5py.File(os.path.join(model_dir, 'preds.h5'))
    test_preds = f['preds'][:]
    f = h5py.File(os.path.join(model_dir, 'targets.h5'))
    test_ground_truth = f['targets'][:]
    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152}

    mask = []
    with open(os.path.join(data_dir, 'sequences.bed'), 'r') as f:
        for line in f.readlines():
            parts = line.split()
            if parts[-1] != 'test':
                continue
            mask.append(0 if parts[0] in ['chrX', 'chrY'] else 1)
    mask = np.array(mask)

    evaluate_variants(test_preds, test_ground_truth, mask, os.path.join(data_dir, 'labels'))
