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
edf_bipolar_path = 'D:\\Maya\\p%s\\P%s_bipolar.edf'
stim_path = 'D:\\Maya\\p%s\\p%s_stim_timing.csv'
scoring_path = 'D:\\Maya\\p%s\\p%s_sleep_scoring.m'
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


def sleep_annotations(subjects):
    for subj in subjects:
        f = sio.loadmat(scoring_path % (subj, subj))
        data = f['sleep_score']
        scoring = np.array(data)[0]
        subj_blocks = []
        stim_sections_sec = get_stim_starts(subj)
        stim_start_sec = stim_sections_sec[0][0]

        # get the 5 minutes before first stim spikes rate (or until the first stim if shorter)
        if stim_start_sec - 60 * 5 < 0:
            start = 0
        else:
            start = stim_start_sec - 60 * 5
        block_scoring = scoring[round(start * 1000): round(stim_start_sec * 1000)]
        block_nrem = np.where(block_scoring == 1)[0]
        sleep_percent = round((len(block_nrem) / len(block_scoring)) * 100, 2)
        subj_blocks.append([start, stim_start_sec, sleep_percent])

        # fill sections of stim and the stops between
        for i, (start, end) in enumerate(stim_sections_sec):
            # fill the current stim
            block_scoring = scoring[round(start * 1000): round(end * 1000)]
            block_nrem = np.where(block_scoring == 1)[0]
            sleep_percent = round((len(block_nrem) / len(block_scoring)) * 100, 2)
            subj_blocks.append([start, end, sleep_percent])
            if i + 1 < len(stim_sections_sec):
                # the stop is the time between the end of the curr section and the start of the next, buffer of 0.5 sec of the stim
                start = end
                end = stim_sections_sec[i + 1][0]
            else:
                # 5 min after the last stim
                start = end
                end = end + 60 * 5
            block_scoring = scoring[round(start * 1000): round(end * 1000)]
            block_nrem = np.where(block_scoring == 1)[0]
            sleep_percent = round((len(block_nrem) / len(block_scoring)) * 100, 2)
            subj_blocks.append([start, end, sleep_percent])

        pd.DataFrame(subj_blocks, columns=['start', 'end', 'sleep_percent']).to_csv(f'results/{subj}_sleep_percent.csv')

# sleep_annotations(['496', '497', '498', '499', '505', '510-1', '510-7', '515', '520', '538', '541', '544', '545'])

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


def plot_bipolar():
    from mat_to_edf import rotem_write_edf
    from mnelab.io.writers import write_edf
    done = ['485', '486', '487', '488', '489', '490', '496', '497', '498', '499', '505', '510-1', '510-7']
    subjects = ['515', '520', '538', '541', '544', '545']
    for subj in subjects:
        print(f'subj: {subj}')
        subj_raw = mne.io.read_raw_edf(edf_path % (subj, subj))
        all_channels = get_clean_channels(subj, subj_raw)
        chans_bi = []

        # get the channels for bipolar reference
        for i, chan in enumerate(all_channels):
            if i + 1 < len(all_channels):
                next_chan = all_channels[i + 1]
                # check that its the same contact
                if next_chan[:-1] == chan[:-1]:
                    chans_bi.append([chan, next_chan])

        # for cleaning
        subj_raw.load_data()
        raw_bi = mne.set_bipolar_reference(subj_raw, [x[0] for x in chans_bi], [x[1] for x in chans_bi], drop_refs=True)

        write_edf(edf_bipolar_path % (subj, subj), raw_bi)
        # raw_bi.plot(duration=30, scalings='auto')


def avg_block_size(subjects):
    rates = {subj: [[], []] for subj in subjects}
    for subj in subjects:
        subj_files_list = [file for file in glob.glob(f'results\\{subj}*rates*') if 'stim' not in file]
        chan_rates = pd.read_csv(subj_files_list[0], index_col=0)
        for i in range(1, len(chan_rates) - 1, 2):
            rates[subj][0].append(chan_rates['duration_sec'].tolist()[i])
            rates[subj][1].append(chan_rates['duration_sec'].tolist()[i + 1])

    stim_means = [round(np.array(rates[subj][0]).mean()) for subj in subjects]
    undisturbed_means = [round(np.array(rates[subj][1]).mean()) for subj in subjects]

    x = np.arange(len(subjects))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, stim_means, width, label='stim', color='tab:orange', alpha=0.5)
    rects2 = ax.bar(x + width / 2, undisturbed_means, width, label='undisturbed', color='tab:blue', alpha=0.5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('duration (sec)')
    ax.set_title('Blocks duration')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend()

    ax.bar_label(rects1, padding=3, fontsize=8)
    ax.bar_label(rects2, padding=3, fontsize=8)
    plt.show()

    # Add scatter
    for i, subj in enumerate(subjects):
        ax.scatter([i - 0.25] * len(rates[subj][0]), rates[subj][0], color='tab:orange', s=14)
        ax.scatter([i + 0.25] * len(rates[subj][1]), rates[subj][1], color='tab:blue', s=14)

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(1)

manual_select_14_thresh = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
# avg_block_size(all_subj)