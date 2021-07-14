# -*- coding: utf-8 -*-
"""
Produce event list with correct triggers in sasi data,
to use for averaging.
"""

from mnefun import extract_expyfun_events, get_raw_fnames, get_event_fnames
import mne
import numpy as np
import os.path as op


def parse_list(fname):
    """Parse a .lst stimulus file into filenames, codes, and isis."""
    with open(fname, 'r') as fid:
        lines = fid.readlines()
    names = list()
    codes = list()
    isis = list()
    while(len(lines) > 0):
        line = lines.pop(0)  # take the next line
        if line.startswith('//'):  # comment, skip
            continue
        # this should be a header line for the given stimulus
        line = line.strip().split()  # split it on whitespace
        assert len(line) == 10  # must have 8 parts
        # distribute to easier to understand name
        # XXX "balance" in the comment seems to store the line count!
        name, code, duration, isi, iti, balance, volume, count, x, xx = line
        assert int(x) == 0  # No idea what these last two entries are...
        assert int(xx) == 6
        count = int(count)
        assert count in (6, 7, 8)  # 7 or 8 parts
        assert int(volume) == 0  # ???
        names.append(name)
        isis.append(int(isi) / 1000.)  # to seconds
        assert isis[-1] == 2.  # might not always be true, but seems to be
        codes.append([[int(code), int(duration)]])
        assert codes[-1][0][0] in (10, 20, 30, 40, 50)
        for ci in range(count):
            line = lines.pop(0).strip().split()
            assert len(line) == 2
            codes[-1].append([int(line[0]), int(line[1]) / 1000.])  # ms->sec
        assert len(codes[-1]) == count + 1
    assert len(names) == len(codes) == len(isis)
    return names, codes, isis


def score(p, subjects, run_indices):
    for si, subj in enumerate(subjects):
        process_subject(p, subj, run_indices[si])


def process_subject(p, subj, ridx):
    if subj in [
            'sasi_117', 'sasi_121', 'sasi_129', 'sasi_131', 'sasi_133',
            'sasi_135', 'sasi_137', 'sasi_143', 'sasi_145']:
        list_fname = 'sentnew2a_FishNew.lst'
    else:
        assert subj in [
            'sasi_110', 'sasi_114', 'sasi_118', 'sasi_120', 'sasi_130',
            'sasi_134', 'sasi_141', 'sasi_144', 'sasi_147'], subj
        list_fname = 'sentnew2b_FishNew.lst'
    list_info = parse_list(list_fname)
    list_info_temp = list_info
    raw_fnames = get_raw_fnames(p, subj, 'raw', False, False, ridx)
    eve_fnames = get_event_fnames(p, subj, ridx)
    assert len(raw_fnames) == len(eve_fnames) == 1
    raw_fname, eve_fname = raw_fnames[0], eve_fnames[0]
    del raw_fnames, eve_fnames

    sentnew2_events, _r = extract_expyfun_events(raw_fname)[:2]
    # Format ids
    sentnew2_events[:, 2] += 10
    sentnew2_events_offset = np.zeros([1500,3], dtype=int)
    sentnew2_critical = np.zeros([1500,3], dtype=int)

    fname_out = op.join(subj, 'lists', f'orig_events_{subj}-eve.lst')
    mne.write_events(fname_out, sentnew2_events)

# NEXT : loop through list_info for each sentence & add offset to timestamp
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

    fname_out_sentnew2 = op.join(
        subj, 'lists', f'offset_events_{subj}-eve.lst')

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
    sentnew2_critical = sentnew2_critical[:stim_count]

# Add info about sentence onset for noise cov
    X = np.append(sentnew2_events, sentnew2_critical, axis=0)
    b = X[X[:, 0].argsort()]
    mne.write_events(eve_fname, b)


#    fname_out_offset = op.join(data_path, '%s' %i, 'lists', 'ALL_%s-eve.lst' %i)
#    mne.write_events(fname_out_offset, X)
