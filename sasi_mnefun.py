# -*- coding: utf-8 -*-

"""
Created on Fri May 6 7:11:32 2016

@author: mdclarke

mnefun script for SASI analysis

"""
import mnefun
import numpy as np
from score import score

### Notes: 
##NO JabGram = 135 
##ECG_channel = 'MEG1531'/EOG channel:
# 120, 121, 129, 137, 141, 143, 144, 147
##ECG_channel = 'MEG1531'/NO EOG channel:
# 110, 114, 117, 118, 135 + 131, 145
##ECG_channel = 'MEG0143'/NO EOG channel
# 130
##No heart artifact in sss data/NO EOG channel
# 133, 134

## 120 Only 8/1076 good ECG epochs found (bad coils)
## 135 Only 12/1076 good ECG epochs found (bad coils)

params = mnefun.Params(tmin=-0.1, tmax=1.2, n_jobs=18,
                       decim=2, proj_sfreq=200, n_jobs_fir='cuda',
                       filter_length='5s', lp_cut=80., n_jobs_resample='cuda',
                       bmin=-0.1, bem_type='5120', ecg_channel='MEG1531')
#
params.subjects = ['sasi_110', 'sasi_114', 'sasi_117', 'sasi_118',
                   'sasi_120', 'sasi_121', 'sasi_129', 'sasi_130', 
                   'sasi_131', 'sasi_133', 'sasi_134', 'sasi_135', 
                   'sasi_137', 'sasi_141', 'sasi_143', 'sasi_144', 
                   'sasi_145', 'sasi_147']

params.structurals = params.subjects
params.dates = [(2013, 0, 00)] * len(params.subjects)
params.subject_indices = np.setdiff1d(np.arange(len(params.subjects)), []) 
#params.subject_indices = []
params.acq_ssh = 'maggie@minea.ilabs.uw.edu'
params.acq_dir = '/sinuhe/data02/sasi'
params.sws_ssh = 'mdclarke@kasga.ilabs.uw.edu'
params.sws_dir = '/data07/maggie/sasi/'
# SSS options
params.sss_type = 'python'
params.sss_regularize = 'in'
params.tsss_dur = 4.
params.int_order = 8
params.st_correlation = .98
params.trans_to='twa'
#params.trans_to='/storage/Maggie/sasi/sasi_mean_trans_n17.fif'
params.coil_t_window = 'auto'
params.movecomp='inter'
# remove segments with < 3 good coils for at least 100 ms
params.coil_bad_count_duration_limit = 0.1
# Trial rejection criteria
params.reject = dict()
params.auto_bad_reject = None
params.ssp_ecg_reject = None
params.flat = dict(grad=1e-13, mag=1e-15)
params.auto_bad_flat = None
params.auto_bad_meg_thresh = 10
params.run_names = ['%s']
params.get_projs_from = np.arange(1)
params.inv_names = ['%s']
params.inv_runs = [np.arange(1)]
params.runs_empty = []
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [1, 1, 0],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)
params.pick_events_cov = lambda x: x[x[:, 2] == 100] # use sentence onset for noise cov
params.cov_method = 'empirical'
params.bem_type = '5120'
params.compute_rank = True
# Epoching
params.in_names = ['EngGram', 'EngUngram', 'JabGram', 'JabUngram', 'Filler']
params.compute_rank = True
params.in_numbers = [12, 13, 14, 15, 16]
params.analyses = ['All',
                   'Conditions',
                   'English',
                   'Jab']
params.out_names = [['All'],
                    ['EngGram', 'EngUngram', 'JabGram', 'JabUngram', 'Filler'],
                    ['EngGram', 'EngUngram'],
                    ['JabGram', 'JabUngram']]
params.out_numbers = [[1, 1, 1, 1, 1],  # Combine all trials
                      [1, 2, 3, 4, 5],
                      [1, 2, -1, -1, -1],
                      [-1, -1, 1, 2, -1]
    ]
params.must_match = [
    [],
    [],
    [0, 1],
    [2, 3]
    ]
ttimes = [0.1, 0.2, 0.4, 0.5, 0.7, 1.0]
params.report_params.update(  # add a couple of nice diagnostic plots
    bem=False,  # Using a surrogate
    whitening=[
        dict(analysis='English', name='EngGram',
             cov='%s-80-sss-cov.fif'),
        dict(analysis='English', name='EngUngram',
             cov='%s-80-sss-cov.fif'),
        dict(analysis='Jab', name='JabGram',
             cov='%s-80-sss-cov.fif'),
        dict(analysis='Jab', name='JabUngram',
             cov='%s-80-sss-cov.fif'),
    ],
    sensor=[
        dict(analysis='English', name='EngGram', times=ttimes),
        dict(analysis='English', name='EngUngram', times=ttimes),
        dict(analysis='Jab', name='JabGram', times=ttimes),
        dict(analysis='Jab', name='JabUngram', times=ttimes),
    ],
    source=[
        dict(analysis='English', name='EngGram',
             inv='%s-80-sss-meg-free-inv.fif', times=ttimes,
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='English', name='EngUngram',
             inv='%s-80-sss-meg-free-inv.fif', times=ttimes,
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='Jab', name='JabGram',
             inv='%s-80-sss-meg-free-inv.fif', times=ttimes,
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='Jab', name='JabUngram',
             inv='%s-80-sss-meg-free-inv.fif', times=ttimes,
             views=['lat', 'caudal'], size=(800, 800)),
    ],
    snr=[
        dict(analysis='English', name='EngGram',
             inv='%s-80-sss-meg-free-inv.fif'),
        dict(analysis='English', name='EngUngram',
             inv='%s-80-sss-meg-free-inv.fif'),
        dict(analysis='Jab', name='JabGram',
             inv='%s-80-sss-meg-free-inv.fif'),
        dict(analysis='Jab', name='JabUngram',
             inv='%s-80-sss-meg-free-inv.fif')
    ],
    psd=False,
)

mnefun.do_processing(
    params,
    fetch_raw=False,
    push_raw=False,
    do_sss=False,
    fetch_sss=False,
    do_ch_fix=False,
    gen_ssp=False,
    apply_ssp=False,
    write_epochs=False,
    gen_covs=False,
    gen_fwd=False,
    gen_inv=False,
    gen_report=True,
    print_status=True
)
