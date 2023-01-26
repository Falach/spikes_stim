import PIL, math
import mne
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

sr = 1000
edf_path = 'D:\\Maya\\p%s\\P%s_fixed.edf'
stim_path = 'D:\\Maya\\p%s\\p%s_stim_timing.csv'
scoring_path = 'D:\\Maya\\p%s\\p%s_sleep_scoring.m'
montage_path = 'D:\\Maya\\p%s\\MacroMontage.mat'
all_subj = ['485', '486', '487', '488', '489', '490', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '520',
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
    start_round = start * 1000 if start * 1000 in stim else round(start * 1000)
    end_round = end * 1000 if end * 1000 in stim else round(end * 1000)
    current = stim[len(stim) - stim[::-1].index(start_round) - 1: stim.index(end_round) + 1]
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

    # final = mne.concatenate_raws(raws[:-1])
    final = mne.concatenate_raws(raws)
    # final.plot(duration=30)
    return final

# def generate_file_name(func, subj, norm, chans):


def sleep_scoring(subjects):
    for subj in subjects:
        f = sio.loadmat(scoring_path % (subj, subj))
        data = f['sleep_score']
        scoring = np.array(data)[0]
        nrem = np.where(scoring == 1)
        plt.plot(scoring)
        stims = get_stim_starts(subj)
        for (start, end) in stims:
            plt.axvspan(start * 1000, end * 1000, facecolor='silver', alpha=0.5)

        total_stim_time = round(1000 * (max(stims)[1] - min(stims)[0]))
        stim_scoring = scoring[round(min(stims)[0] * 1000): round(max(stims)[1] * 1000)]
        stim_nrem = np.where(stim_scoring == 1)[0]
        sleep_percent = round(len(stim_nrem) / total_stim_time * 100, 2)
        plt.title(f'{subj} scoring and stim sessions- {sleep_percent}% sleep')
        plt.xlabel('Time')
        plt.ylabel('scoring value')
        plt.xlim(0, len(scoring))
        # plt.ylim(-1, 1)
        plt.locator_params(axis='y', nbins=3)
        plt.savefig(f'results/scoring/{subj}_scoring_and_stim.png')
        plt.clf()


def get_clean_channels(subj, raw):
    to_remove = ['C3', 'C4', 'PZ', 'CZ', 'EZ', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'A1', 'A2']
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

# sleep_scoring(all_subj)
# for x in all_subj:
#     get_stim_starts(x)