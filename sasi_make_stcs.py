#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:30:07 2017

@author: mdclarke
"""
### create source time course files for conditions

# 1) mne_watershed_bem --subject subj
# 2) rename surfaces ex: inner_skull.surf and put in bem folder
# 3) mne_make_scalp_surfaces --subject subj
# 4) coreg
# 6) run gen fwd and gen inv in sasi_mnefun.py
###############################################################################

import os
import mne
import numpy as np
from os import path as op
from mne.minimum_norm import (make_inverse_operator, apply_inverse, 
                              write_inverse_operator, read_inverse_operator)

subj = ['sasi_110', 'sasi_114', 'sasi_117', 'sasi_118',
        'sasi_120', 'sasi_121', 'sasi_129', 'sasi_130', 
        'sasi_131', 'sasi_133', 'sasi_134', 'sasi_135',
        'sasi_137', 'sasi_141', 'sasi_143', 'sasi_144', 
        'sasi_145', 'sasi_147']
data_path = '/storage/Maggie/sasi'
anat_path = '/storage/Maggie/anat/subjects'
conditions = ['EngGram', 'EngUngram', 'JabGram', 'JabUngram']
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2

source = mne.read_source_spaces(op.join(anat_path, 'BBC_249', 'bem',
                                        'BBC_249-oct-6-src.fif'))
source_verts = [source[0]['vertno'], source[1]['vertno']]
#source_verts = [np.arange(10242), np.arange(10242)]
################ HERE WE GO ###################################################
for si, s in enumerate(subj):    
    fwd = mne.read_forward_solution(op.join(data_path, '%s' % s, 'forward',
                                    '%s-sss-fwd.fif' % s))
    cov = mne.read_cov(op.join(data_path, '%s' % s, 'covariance',
                                    '%s-80-sss-cov.fif' % s))
    inv = read_inverse_operator(op.join(data_path, '%s' % s, 'inverse',
                                    '%s-80-sss-meg-free-inv.fif' % s))
    evokeds = [mne.read_evokeds(op.join(data_path, '%s' %s,
                                       'inverse', 'Conditions_80-sss_eq_%s-ave.fif' % s),
                                condition=c) for c in conditions]
    stcs = [apply_inverse(e, inv, lambda2, method=method, 
                          pick_ori=None) for e in evokeds]
    if not op.isdir(op.join(data_path, '%s' %s, 'stc')):
        os.mkdir(op.join(data_path, '%s' %s, 'stc'))
    for j, stc in enumerate(stcs):
        stc.save(op.join(data_path, '%s' %s, 'stc',
                         '%s_' % s + evokeds[j].comment)) 
    m = mne.compute_morph_matrix(s, 'BBC_249', stcs[0].vertices,
                                 source_verts)
    morphed_stcs = [stcz.morph_precomputed('BBC_249-5-src.fif', 
                                           source_verts, m)for stcz in stcs]
    for j, stc in enumerate(morphed_stcs):
        stc.save(op.join(data_path, '%s' %s, 'stc',
                         '%s_BBC_249_morph_' % s + evokeds[j].comment))
    if si == 0:
        data=np.empty((len(morphed_stcs), len(subj), morphed_stcs[0].shape[0], 
                       morphed_stcs[0].shape[1])) # conditions X subjs X space X time
    for k, stc in enumerate(morphed_stcs):
        data[k, si] = stc.data
#    test = mne.SourceEstimate(data[0].squeeze(), vertices=source_verts,
#                              tmin=-.2, tstep=np.diff(stcs[0].times[:2]),
#                              subject='BBC_249')
#    test.save(op.join(data_path, '%s' %s, 'stc', '%s_' %s + evokeds[j].comment))
