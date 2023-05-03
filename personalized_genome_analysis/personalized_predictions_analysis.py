import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tracks(tracks, interval, height=1.5, save_path='out.png'):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    if len(tracks) == 1:
        ax = axes
        items = list(tracks.items())
        ax.fill_between(np.linspace(0, 896, num=len(items[0][1])), items[0][1])
        ax.set_title(items[0][0])
        sns.despine(top=True, right=True, bottom=True)
    else:
        for ax, (title, y) in zip(axes, tracks.items()):
            ax.fill_between(np.linspace(0, 896, num=len(y)), y)
            ax.set_title(title)
            sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close(fig)


def evaluate(test_predictions, test_gt, output_dir):
    # evaluate
    for i in range(test_predictions.shape[2]):
        tracks = {f'F1 pred - seq {100}': test_predictions[100, :, i],
                  f'F1 gt - seq {100}': test_gt[100, :, i]}
        plot_tracks(tracks, 'int', save_path=os.path.join(output_dir, f'tracks_{i}.png'))


if __name__ == '__main__':
    prediction_dir = '/g/furlong/project/103_Basenji/analysis/personalised_genomes/'
    important_tracks = {'twi.24': 12, 'bin.1012': 94, 'ctcf.68': 271, 'mef2.1012': 467, 'mef2.68': 628, 'bin.68': 1152}
    for dir_ind in ['DGRP-28', 'DGRP-307', 'DGRP-399', 'DGRP-57', 'DGRP-639', 'DGRP-712', 'DGRP-714', 'DGRP-852']:
        f = h5py.File(os.path.join(prediction_dir, dir_ind, 'predict.h5'), 'r')
        test_preds = f['preds'][:]
        test_preds = test_preds[:, :, list(important_tracks.values())]
        gt_folder = rf'/g/furlong/project/103_Basenji/data/personalised_predictions/{dir_ind}/'
        test_gt = []
        for tf in important_tracks.keys():
            f = h5py.File(os.path.join(gt_folder, f'{tf}_targets.h5'), 'r')
            test_gt.append(f['targets'][:])
        test_gt = np.nan_to_num(np.array(test_gt).astype('float16'))
        test_gt = np.rollaxis(test_gt, 0, test_gt.ndim)
        evaluate(test_preds, test_gt, os.path.join(prediction_dir, dir_ind))
        os.makedirs(os.path.join('/scratch/stojanov/basenji/experiments/personalized_genome_analysis/', dir_ind),
                    exist_ok=True)
        evaluate(test_preds, test_gt,
                 os.path.join('/scratch/stojanov/basenji/experiments/personalized_genome_analysis/', dir_ind))
