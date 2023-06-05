import PIL, math
import mne
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import glob

sr = 1000
edf_path = 'D:\\Maya\\p%s\\P%s_fixed.edf'
control_path = 'C:\\UCLA\\P%s_overnightData.edf'
edf_bipolar_path = 'D:\\Maya\\p%s\\P%s_bipolar.edf'
stim_path = 'D:\\Maya\\p%s\\p%s_stim_timing.csv'
scoring_path = 'D:\\Maya\\p%s\\p%s_sleep_scoring.m'
control_scoring_path = 'D:\\UCLA_NREM\\P%s_hypno.txt'
montage_path = 'D:\\Maya\\p%s\\MacroMontage.mat'
all_subj = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '520',
            '538', '541', '544', '545']

# get the blocks start and end time
def get_stim_starts(subj):
    stim = np.array(pd.read_csv(stim_path % (subj, subj), header=None).iloc[0, :])
    stim_sessions = []
    start = stim[0] / 1000
    end = None
    for (i, x) in enumerate(stim):
        if end is not None:
            start = stim[i] / 1000
            end = None
        # Check if the next stim is in more than 5 minutes
        if i + 1 < stim.size and stim[i + 1] - stim[i] > 5 * 60 * 1000:
            end = stim[i] / 1000
            # check that it isn't a single stim (like 487, 9595 sec) or shorter than 1 min (like 497)
            if stim[i] / 1000 - start > 60:
                stim_sessions.append((start, end))
            else:
                print(subj)

    return stim_sessions

# TODO: 497 check padding- first and last sec of block
def remove_stim(subj, block_raw, start, end):
    # stim = np.array(pd.read_csv(stim_path % (subj, subj), header=None).iloc[0, :])
    raws = []
    stim = pd.read_csv(stim_path % (subj, subj), header=None).iloc[0, :].to_list()
    stim = [round(x) for x in stim]
    # get the relevant section from the list with all stimuli
    start_stim = [x for x in stim if x >= start * 1000]
    end_stim = [x for x in stim if x <= end * 1000]
    if start_stim == [] or end_stim == [] or start_stim[0] >= end_stim[-1]:
        return block_raw
    else:
        start_stim = start_stim[0]
        end_stim = end_stim[-1]
    current = stim[len(stim) - stim[::-1].index(start_stim) - 1: stim.index(end_stim) + 1]
    for i, stim_time in enumerate(current):
        if i + 1 < len(current):
            tmin = current[i] / 1000 - start + 0.5
            tmax = current[i + 1] / 1000 - start - 0.5
            if tmax > tmin:
                curr_raw = block_raw.copy().crop(tmin=tmin, tmax=tmax)
                raws.append(curr_raw)
                # TODO: add flat on 1 sec?
                # last_vals = block_raw.get_data()[:, -1]
                # info = mne.create_info(ch_names=block_raw.ch_names, sfreq=sr)
                # flat_1 = np.full(60 * sr, last_vals[0])
                # flat_2 = np.full(60 * sr, last_vals[1])
                # raws.append(mne.io.RawArray(np.vstack((flat_1, flat_2)), info))

    return mne.concatenate_raws(raws)


def get_clean_channels(subj, raw):
    to_remove = ['C3', 'C4', 'PZ', 'CZ', 'EZ', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'A1', 'A2']
    specific = {'499': ['RSTG6', 'RSTG7', 'RSMG7'], '510-7': ['LA8', 'LOF8', 'RIA7']}
    if subj in specific.keys():
        to_remove.extend(specific[subj])
    # doesn't have bad channels
    if subj != '520':
        f = sio.loadmat(montage_path % subj)
        montage = np.array(f['MacroMontage'][0])
        chan_index = 1
        last_chan = str(montage[0][1][0])
        for chan in montage:
            if len(chan['Area']) > 0 and str(chan['Area'][0]) != last_chan:
                chan_index = 1
                last_chan = str(chan['Area'][0])
            # remove if there is a flag of bad channel
            if chan['badChannel'] in [1, 2, 5]:
                to_remove.append(str(chan['Area'][0]) + str(chan_index))
            chan_index += 1

    final = [chan for chan in raw.ch_names if chan.upper() not in to_remove]

    return final

def get_control_clean_channels(subj, raw):
    to_remove = ['C3', 'C4', 'PZ', 'CZ', 'EZ', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'A1', 'A2', 'X1', 'X1-REF', 'ECG']
    final = [chan for chan in raw.ch_names if chan.upper() not in to_remove]
    return final


def get_nrem_epochs(subj='485', with_stim=True):
    f = sio.loadmat(scoring_path % (subj, subj))
    data = f['sleep_score']
    scoring = np.array(data)[0]
    nrem = np.where(scoring == 1)
    nrem_epochs = []
    start = nrem[0][0]
    for i in range(len(nrem[0]) - 1):
        if nrem[0][i + 1] - nrem[0][i] != 1:
            nrem_epochs.append([start / 1000, nrem[0][i] / 1000])
            start = nrem[0][i + 1]

    # in case there is only one epoch
    if nrem_epochs == []:
        nrem_epochs.append([start / 1000, nrem[0][-1] / 1000])

    if with_stim:
        nrem_stim_epochs = []
        stim = get_stim_starts(subj)
        for epoch in nrem_epochs[:]:
            for stim_start in stim:
                if epoch[0] <= stim_start[0] <= epoch[1]:
                    nrem_stim_epochs.append(epoch)
                    # nrem_epochs.remove(epoch)
                    break

    return nrem_epochs, nrem_stim_epochs, stim

def is_stim_in_epoch(subj, start, end):
    stim = get_stim_starts(subj)
    for stim_start in stim:
        if start <= stim_start[0] <= end:
            return True

    return False

def get_control_nrem_epochs(subj='402'):
    scoring = np.loadtxt(control_scoring_path % subj)
    # n1 =1, n2 =2, n3 =3
    nrem = np.where(np.logical_and(scoring >= 1, scoring <= 3))
    nrem_epochs = []
    start = nrem[0][0]
    for i in range(len(nrem[0]) - 1):
        if nrem[0][i + 1] - nrem[0][i] != 1:
            # each epoch is 30 sec
            nrem_epochs.append([start * 30, nrem[0][i] * 30])
            start = nrem[0][i + 1]

    return nrem_epochs


manual_select_14_thresh = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
# avg_block_size(all_subj)
# plot_stim_duration(all_subj)
# get_control_nrem_epochs()