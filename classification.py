# -*- coding: utf-8 -*-
"""
Do machine learning on conditions.
"""

import os
import os.path as op
import time

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import mne

subjects = [110, 114, 117, 118, 121,  # 120 is missing JabGram
            129, 130, 131, 133, 134, 137,
            141, 143, 144, 145, 147]  # 135 has no jabber
subjects = ['sasi_%d' % subj for subj in subjects]
subjects_dir = '/storage/Maggie/anat/subjects/'
decim = 10
os.makedirs('classification', exist_ok=True)

contrasts = dict(
    evj=['EngGram', 'JabGram'],
    eng=['EngGram', 'EngUngram'],
    jab=['JabGram', 'JabUngram'],
)
ignore_conditions = ['Filler']

all_scores = all_patterns = None
for si, subject in enumerate(subjects):
    out_fname = op.join('classification', f'{subject}-scores.h5')
    if op.isfile(out_fname):
        data = mne.externals.h5io.read_hdf5(out_fname)
        if all_scores is None:
            all_scores = np.zeros((len(subjects),) + data['scores'].shape)
            all_patterns = np.zeros((len(subjects),) + data['patterns'].shape)
        all_scores[si] = data['scores']
        all_patterns[si] = data['patterns']
        continue
    print(f'{subject}')
    epochs = mne.read_epochs(op.join(
        subject, 'epochs', 'All_80-sss_%s-epo.fif' % subject))
    epochs.load_data().pick_types(meg=True).filter(None, 10., verbose='error')
    assert len(epochs.ch_names) == 306
    clf = make_pipeline(
        mne.decoding.Scaler(epochs.info),
        mne.decoding.Vectorizer(),
        PCA(0.9999, whiten=True),
        mne.decoding.LinearModel(
            LogisticRegression(C=1., random_state=0, max_iter=1000,
                               solver='lbfgs')),
    )
    time_gen = mne.decoding.GeneralizingEstimator(
        clf, 'roc_auc', 1, verbose=False)  # balanced_accuracy
    these_scores = these_patterns = None
    for ci, (key, contrast) in enumerate(contrasts.items()):
        print(f'    {key}: ', end='')
        assert len(contrast) == 2
        t0 = time.time()
        these_epochs = epochs[contrast]
        assert these_epochs.info['sfreq'] == 500.
        these_epochs.decimate(decim, verbose='error')  # to 50 Hz
        a, b = these_epochs[contrast[0]], these_epochs[contrast[1]]
        assert 20 < min(len(a), len(b)) < 50, (len(a), len(b))
        mne.epochs.equalize_epoch_counts([a, b])
        these_epochs = mne.concatenate_epochs([a, b])
        assert len(set(these_epochs.events[:, 2])) == 2
        X = these_epochs.get_data()
        y = these_epochs.events[:, 2]
        cv = StratifiedKFold(random_state=0, shuffle=True)
        scores = mne.decoding.cross_val_multiscore(
            time_gen, X, y, cv=cv, n_jobs=1, verbose=False)
        if these_scores is None:
            these_scores = np.zeros((len(contrasts),) + scores.shape)
            these_patterns = np.zeros((len(contrasts),) + X.shape[1:])
        these_scores[ci] = scores
        med, hi = np.percentile(np.diag(scores.mean(0)), [50, 90])
        print(
            f' {100 * hi:0.1f}% '
            f'(med: {100 * med:0.1f}%%, {time.time() - t0:0.1f} sec)')
        clf.fit(X, y)
        these_patterns[ci] = mne.decoding.get_coef(
            clf, 'patterns_', inverse_transform=True)
    mne.externals.h5io.write_hdf5(
        out_fname, dict(scores=these_scores, patterns=these_patterns))
    if all_scores is None:
        all_scores = np.zeros((len(subjects),) + these_scores.shape)
        all_patterns = np.zeros((len(subjects),) + these_patterns.shape)
    all_scores[si] = these_scores
    all_patterns[si] = these_patterns


times = mne.read_epochs(
    op.join(subject, 'epochs', 'All_80-sss_%s-epo.fif' % subject)
).decimate(decim, verbose='error').times

n_row, n_col = len(contrasts), 1
fig, axes = plt.subplots(
    n_row, n_col, figsize=(n_col * 6, n_row * 3), constrained_layout=True)
for ki, key in enumerate(contrasts):
    ax = axes[ki]
    scores = np.diagonal(all_scores[:, ki].mean(1), axis1=1, axis2=2)
    ax.plot(times, np.mean(scores, 0), color='k', lw=2, zorder=5)
    ax.plot(times, scores.T, color='k', alpha=0.25, lw=1, zorder=4)
    ax.axhline(0.5, color='k', ls='--', lw=1., zorder=3)
    ax.set(xlim=times[[0, -1]], ylim=[0.25, 0.75], ylabel=f'{key} AUC')
    ax.grid(True, ls=':', color='0.5')
    if ki == len(contrasts) - 1:
        ax.set(xlabel='Time (sec)')
fig.savefig(op.join('classification', 'classification.png'))
