# -*- coding: utf-8 -*-
"""
script to produce event list with correct triggers in sasi data, 
to use for averaging. Adjust script for a or b and subject group
"""

from process_sasi_list import parse_list
from mnefun import extract_expyfun_events
import mne
import numpy as np
import os.path as op

list_type = 'B' # A or B
data_path = '/storage/Maggie/sasi/'

if list_type == 'A':
         list_fname = data_path + 'sentnew2a_FishNew.lst'
         subjs = ['sasi_117', 'sasi_121', 'sasi_129', 'sasi_131', 
                  'sasi_133', 'sasi_135', 'sasi_137', 'sasi_143',
                  'sasi_145']
if list_type == 'B':
         list_fname = data_path + 'sentnew2b_FishNew.lst'
         subjs = ['sasi_110', 'sasi_114', 'sasi_118', 'sasi_120',
                  'sasi_130', 'sasi_134', 'sasi_141', 'sasi_144',
                  'sasi_147']
else:
         print("list_type must be A or B")

list_info = parse_list(list_fname)
list_info_temp = list_info

for i in subjs:
    raw_fname = op.join(data_path, '%s' %i, 'raw_fif', '%s_raw.fif' %i)

    sentnew2_events, _r = extract_expyfun_events(raw_fname)[:2]
    # Format ids
    sentnew2_events[:, 2] += 10
    sentnew2_events_offset = np.zeros([1500,3], dtype=int)
    sentnew2_critical = np.zeros([1500,3], dtype=int)

    fname_out = op.join(data_path, '%s' %i, 'lists', 'orig_events_%s-eve.lst'
                        %i)
    mne.write_events(fname_out, sentnew2_events)

# Loop through list_info for each sentence & add offset to timestamp
    num_sentences = (len(sentnew2_events)) - 1
# Loop through sentences
    total_word_count = 0
    last_word = 0
    #for sentence_count in range(0, 3):
    for sentence_count in range(0, 170): 
            last_word = 0
            for word_count in range(0, len(list_info_temp[1][sentence_count])):
                if word_count == 0:
                    # Extract the sentence onset and the event code
                    sentence_onset = sentnew2_events[sentence_count,0] # + int(list_info_temp[1][sentence_count][word_count][1])
                    sentnew2_events_offset[total_word_count][0]= sentence_onset
                    sentnew2_events_offset[total_word_count][2]=sentnew2_events[sentence_count][2] # Event ID
                    last_word = last_word + 1
                    total_word_count = total_word_count + 1
                else: 
                    # Extract all words and times
                    word_offset = int(list_info_temp[1][sentence_count][word_count][1] * 1000)
                    sentnew2_events_offset[total_word_count][0]= word_offset #0+ sentence_onset
                    sentnew2_events_offset[total_word_count][2]= last_word # Which is 1st word?
                    last_word = last_word + 1
                    total_word_count = total_word_count + 1

    fname_out_sentnew2 = op.join(data_path, '%s' %i, 'lists', 
                                 'offset_events_%s-eve.lst' %i)

    mne.write_events(fname_out_sentnew2, sentnew2_events_offset)   

    stim_count = 0
    for n, r in enumerate(sentnew2_events[:,2]):
                if r == 12:
                    sentnew2_events[:,2][n] = 100
                if r == 13:
                    sentnew2_events[:,2][n] = 100
                if r == 14:
                    sentnew2_events[:,2][n] = 100
                if r == 15:
                    sentnew2_events[:,2][n] = 100
                if r == 16:
                    sentnew2_events[:,2][n] = 100
            
    # Extract critical words timing
        # If event = 12,13,14,15 then timing aligns with critical word 
        # rev c then no offset
    for word_count in range(0, len(sentnew2_events_offset)):
        if sentnew2_events_offset[word_count][2] == 12:
            sentnew2_critical[stim_count][2] = 12
            sentnew2_critical[stim_count][0] = sentnew2_events_offset[word_count + 5][0] + sentnew2_events_offset[word_count][0]
            stim_count = stim_count + 1
        if sentnew2_events_offset[word_count][2] == 13:
            sentnew2_critical[stim_count][2] = 13
            sentnew2_critical[stim_count][0] = sentnew2_events_offset[word_count + 3][0] + sentnew2_events_offset[word_count][0]
            stim_count = stim_count + 1
        if sentnew2_events_offset[word_count][2] == 14:
            sentnew2_critical[stim_count][2] = 14
            sentnew2_critical[stim_count][0] = sentnew2_events_offset[word_count + 6][0] + sentnew2_events_offset[word_count][0]
            stim_count = stim_count + 1
        if sentnew2_events_offset[word_count][2] == 15:
            sentnew2_critical[stim_count][2] = 15
            sentnew2_critical[stim_count][0] = sentnew2_events_offset[word_count + 6][0] + sentnew2_events_offset[word_count][0]
            stim_count = stim_count + 1
        if sentnew2_events_offset[word_count][2] == 16:
            sentnew2_critical[stim_count][2] = 16
            sentnew2_critical[stim_count][0] = sentnew2_events_offset[word_count + 1][0] + sentnew2_events_offset[word_count][0]
            stim_count = stim_count + 1
    
# Add info about sentence onset for noise cov       
    X = np.append(sentnew2_events, sentnew2_critical, axis=0)
    b = X[X[:, 0].argsort()]
    fname_out_offset = op.join(data_path, '%s' %i, 'lists', 'ALL_%s-eve.lst'
                               %i)
    mne.write_events(fname_out_offset, b)
            
#    fname_out_offset = op.join(data_path, '%s' %i, 'lists', 'ALL_%s-eve.lst' %i)
#    mne.write_events(fname_out_offset, X)
