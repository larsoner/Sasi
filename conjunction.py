# -*- coding: utf-8 -*-
"""
author: Eric Larson

Create source models and do MNE localization.
Run conjunction and partial conjunction analyses

"""
from __future__ import print_function

import os
import os.path as op
import time

import numpy as np

import mne
from mnefun import discretize_cmap
from mnefun.stats import (hotelling_t2, hotelling_t2_baseline,
                          partial_conjunction)

subjects = [110, 114, 117, 118, 120, 121, 129, 130, 133, 134,
            137, 141, 143, 144, 145]  # 135 has no jabber
subjects = ['sasi_%d' % subj for subj in subjects]
subjects_dir = '/storage/Maggie/anat/subjects/'
path = '/storage/Maggie/sasi/'

smoothing_steps = 5  # when plotting
plot_lims = (0.25, 0.29, 0.33)
con_thresh = 1e-8  # # Consistency map p-value threshold
con_lims = [len(subjects) // 2, len(subjects)]
pc_lims = con_lims  # Partial conjunctions
conditions = ['EngGram', 'EngUngram', 'JabGram', 'JabUngram']
ignore_conditions = ['Filler']
tois = [(0.3, 0.5), (0.5, 0.85)]
lambda2 = 1. / 9.
plot_subj = 'BBC_249'
p_kind = 'baseline'  # or 'time', much slower!
method = 'dSPM'  # inv method to use
inspect = False  # useful for setting up plots
assert method in ('MNE', 'dSPM', 'eLORETA', 'sLORETA')
for d in ('stcs', 'results'):
    if not op.isdir(d):
        os.mkdir(d)
stc_contrast = list()
src = mne.read_source_spaces(
    op.join(subjects_dir, plot_subj, 'bem',
            plot_subj + '-oct-6-src.fif'))
subj_verts = [s['vertno'] for s in src]
del src
for ci, condition in enumerate(conditions):
    stcs = list()
    stc_ps = list()
    print(condition)
    for subject in subjects:
        fname_stc = op.join(path, 'stcs', '%s_%s_%s'
                            % (subject, condition, method.lower()))
        fname_stc_p = op.join(path, 'stcs', '%s_%s_%s_p'
                              % (subject, condition, method.lower()))
        if not all(op.isfile(fname) for fname in
                   (fname_stc + '-stc.h5', fname_stc_p + '-lh.stc')):
            t0 = time.time()
            print('  %s: Epoching' % subject, end='')
            epochs = mne.read_epochs(op.join(path, 
                subject, 'epochs', 'All_80-sss_%s-epo.fif' % subject))
            assert set(epochs.event_id.keys()) == set(conditions +
                                                      ignore_conditions)
            inv = mne.minimum_norm.read_inverse_operator(op.join(path,
                    subject, 'inverse',
                    '%s-80-sss-meg-free-inv.fif' % subject))
            assert inv['src'][0]['subject_his_id'] == subject
            these_verts = [s['vertno'] for s in inv['src']]
            n_ave = len(epochs)
            evoked_op = mne.EvokedArray(np.eye(len(epochs.ch_names)),
                                        epochs.info, tmin=epochs.times[0])
            print(' : %s' % method, end='')
            stc = mne.minimum_norm.apply_inverse(
                evoked_op, inv, lambda2, method, pick_ori='vector')
            assert stc.data.shape[0] == 8196
            assert all(np.array_equal(a['vertno'], b)
                       for a, b in zip(inv['src'], subj_verts))
            inv_op = stc.data
            picks = np.arange(inv_op.shape[-1])
            stc = mne.VectorSourceEstimate(
                np.dot(inv_op, epochs[condition].average(picks=picks).data),
                subj_verts, stc.tmin, stc.tstep, plot_subj)
            stc.save(fname_stc)
            brain = stc.magnitude().plot(
                views='lat', hemi='lh',
                smoothing_steps=smoothing_steps,
                colormap='cool', clim=dict(kind='value', lims=plot_lims),
                subjects_dir=subjects_dir, figure=ci)
            print(' : Hotelling', end='')
            if p_kind == 'baseline':
                stc = hotelling_t2_baseline(stc, n_ave, epochs.baseline,
                                            check_baseline=False)
            else:
                assert p_kind == 'time'
                stc = hotelling_t2(epochs[condition], inv_op)
            stc.save(fname_stc_p)
            del stc, inv_op
            print(' %0.1f sec' % (time.time() - t0,))
        stcs.append(mne.read_source_estimate(fname_stc))
        stc_ps.append(mne.read_source_estimate(fname_stc_p))
    for start, stop in tois:
        # Grand average
        fname = op.join(path, 'results',
                        '%s_%s_%s_%s.png' % (condition, method, start, stop))
        assert np.allclose(stcs[0].times, stc_ps[0].times)
        mask = (stcs[0].times >= start) & (stcs[0].times <= stop)
        assert mask.sum() > 0
        tmin, tstep = stcs[0].tmin, stcs[0].tstep
        if not op.isfile(fname):
            print('Plotting %s ...' % fname)
            gave = sum(stc.magnitude().data for stc in stcs)
            gave /= len(stcs)
            if inspect:
                import matplotlib.pyplot as plt
                plt.plot(stcs[0].times, np.max(gave, axis=0))  # time
            gave = gave[:, mask].mean(-1, keepdims=True)
            if inspect:
                print(np.percentile(gave.ravel(), [90, 95, 99]))
            gave = mne.SourceEstimate(
                gave, subj_verts, start, tstep, plot_subj)
            brain = gave.plot(
                views=['lat', 'med'], hemi='split',
                smoothing_steps=smoothing_steps,
                colormap='cool', clim=dict(kind='value', lims=plot_lims),
                subjects_dir=subjects_dir, time_viewer=inspect)
            if inspect:
                raise RuntimeError
            brain.save_image(fname)
            brain.close()

        # Consistency map
        fname = op.join(path, 'results', '%s_%s_%s_%s_con.png'
                        % (condition, method, start, stop))
        if not op.isfile(fname):
            t0 = time.time()
            print('Plotting %s ...' % fname)
            con = sum((p.data[:, mask] <
                       con_thresh).any(-1, keepdims=True)
                      for p in stc_ps)
            con = mne.SourceEstimate(
                con, subj_verts, start, tstep, plot_subj)
            con.save(path, 'results', '%s_%s_%s_%s_%s_con'
                     % (subject, condition, method, start, stop))
            colormap, use_con_lims = discretize_cmap('viridis_r', con_lims,
                                                     transparent=True)
            brain = con.plot(
                views=['lat', 'med'], hemi='split', colormap=colormap,
                smoothing_steps=smoothing_steps, transparent=False,
                clim=dict(kind='value', lims=use_con_lims),
                subjects_dir=subjects_dir, time_viewer=inspect)
            if inspect:
                raise RuntimeError
            brain.save_image(fname)
            brain.close()

        # Partial conjunction
        fname = op.join(path, 'results', '%s_%s_%s_%s_pc.png'
                        % (condition, method, start, stop))
        if not op.isfile(fname):
            t0 = time.time()
            print('Plotting %s ...' % fname, end='')
            pc = partial_conjunction([p.data[:, mask]
                                      for p in stc_ps])[0]
            initial_time = np.mean([start, stop])
            pc = mne.SourceEstimate(
                pc, subj_verts, stc_ps[0].times[mask][0], tstep, plot_subj)
            pc.save(path, 'results', '%s_%s_%s_%s_%s_con' 
                    % (subject, condition, method, start, stop))
            colormap, use_pc_lims = discretize_cmap('viridis_r', pc_lims,
                                                    transparent=True)
            brain = pc.plot(
                views=['lat', 'med'], hemi='split', colormap=colormap,
                smoothing_steps=smoothing_steps, transparent=True,
                clim=dict(kind='value', lims=use_pc_lims), time_viewer=inspect,
                subjects_dir=subjects_dir, initial_time=initial_time)
            brain.save_image(fname)
            brain.close()
            print(' (%0.1f sec)' % (time.time() - t0,))
