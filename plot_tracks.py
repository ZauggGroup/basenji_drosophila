import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tracks(tracks, labels, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(0, 896, num=len(y)), y, where=labels == 0)
        # ax.fill_between(np.linspace(0, 896, num=len(y)), y, where=labels == 1, facecolor='orange')
        # ax.fill_between(np.linspace(0, 896, num=len(y)), y, where=labels == 2, facecolor='red')
        ax.fill_between(np.linspace(0, 896, num=len(y)), y, where=labels != 0, facecolor='red')
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # data_dir = '/g/scb/zaugg/stojanov/basenji/experiments/data/new_data/drosophila_l131k/'
    data_dir = r'Y:\stojanov\basenji\experiments\data\new_data\drosophila_l131k'
    # model_dir = '/g/scb/zaugg/stojanov/basenji/experiments/models/drosophila_l131k_new_data_augmented'
    model_dir = r'Y:\stojanov\basenji\experiments\models\drosophila_l131k_new_data_augmented'
    f = h5py.File(os.path.join(model_dir, 'preds.h5'))
    test_preds = f['preds'][:]
    f = h5py.File(os.path.join(model_dir, 'targets.h5'))
    test_gt = f['targets'][:]
    automated_peaks = False

    mask = []
    with open(os.path.join(data_dir, 'sequences.bed'), 'r') as f:
        for line in f.readlines():
            parts = line.split()
            if parts[-1] != 'test':
                continue
            mask.append(0 if parts[0] in ['chrX', 'chrY'] else 1)
    mask = np.array(mask)

    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152}
    for label, ind in important_tracks.items():
        variants_labels = np.load(os.path.join(data_dir, 'labels', f'{label}_peaks.npz'))['data']
        test_gt_ti = test_gt[:, :, ind]
        test_preds_ti = test_preds[:, :, ind]
        # test_preds_i = test_preds_i[np.where(mask == 1)]
        tracks = {f'F1 target - seq {0}': test_gt_ti[0, :],
                  f'F1 pred - seq {0}': test_preds_ti[0, :]}
        plot_tracks(tracks, variants_labels[0, :], 'target_interval')
