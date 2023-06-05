import PIL, math
import mne
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
from utils import get_stim_starts, get_clean_channels

sr = 1000
edf_path = 'D:\\Maya\\p%s\\P%s_fixed.edf'
control_path = 'C:\\UCLA\\P%s_overnightData.edf'
edf_bipolar_path = 'D:\\Maya\\p%s\\P%s_bipolar.edf'
stim_path = 'D:\\Maya\\p%s\\p%s_stim_timing.csv'
scoring_path = 'D:\\Maya\\p%s\\p%s_sleep_scoring.m'
montage_path = 'D:\\Maya\\p%s\\MacroMontage.mat'
all_subj = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '520',
            '538', '541', '544', '545']

def sleep_scoring(subjects):
    for subj in subjects:
        f = sio.loadmat(scoring_path % (subj, subj))
        data = f['sleep_score']
        scoring = np.array(data)[0]
        nrem = np.where(scoring == 1)
        plt.step([x / sr / 60 / 60 for x in range(len(scoring))], scoring)
        stims = get_stim_starts(subj)
        for (start, end) in stims:
            plt.axvspan(start / 60 / 60, end / 60 / 60, facecolor='silver', alpha=0.5)

        total_stim_time = round(1000 * (max(stims)[1] - min(stims)[0]))
        stim_scoring = scoring[round(min(stims)[0] * 1000): round(max(stims)[1] * 1000)]
        stim_nrem = np.where(stim_scoring == 1)[0]
        sleep_percent = round(len(stim_nrem) / total_stim_time * 100, 2)
        plt.title(f'{subj} scoring and stim sessions- {sleep_percent}% sleep')
        plt.xlabel('Time (h)')
        plt.ylabel('scoring value')
        plt.xticks(np.arange(len(scoring) / sr / 60 / 60, step=1))
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


def plot_stim_duration(subjects):
    rates = {}
    for subj in subjects:
        subj_files_list = [file for file in glob.glob(f'results\\{subj}*rates*') if 'stim' not in file]
        chan_rates = pd.read_csv(subj_files_list[0], index_col=0)
        # minus 10 min for the baseline and end of the session
        rates[subj] = (sum(chan_rates['duration_sec'].tolist()) - 600) / 60

    rates_mean = np.array(list(rates.values())).mean()
    x = np.arange(len(subjects))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, rates.values(), width, alpha=0.5)
    ax.axhline(y=rates_mean, color='r', linestyle='-', label='mean')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('duration (min)')
    ax.set_title('total stim session duration')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print(1)


def plot_before_stim_duration(subjects):
    rates = {}
    for subj in subjects:
        subj_files_list = [file for file in glob.glob(f'results\\{subj}_*_nrem_split_rates.csv')]
        if len(subj_files_list) > 0 :
            chan_rates = pd.read_csv(subj_files_list[0], index_col=0)
            before_stim = chan_rates[chan_rates.is_stim == 0]
            total_duration = before_stim['duration_sec'].sum() / 60
            rates[subj] = total_duration

    # bug fix
    del rates['497']
    del rates['515']
    del rates['545']
    rates_mean = np.array(list(rates.values())).mean()
    x = np.arange(len(rates.keys()))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, rates.values(), width, alpha=0.5)
    ax.axhline(y=rates_mean, color='r', linestyle='-', label='mean')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('duration (min)')
    ax.set_title('total rem before stim')
    ax.set_xticks(x)
    ax.set_xticklabels(rates.keys())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


manual_select_14_thresh = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
# avg_block_size(all_subj)
# plot_stim_duration(all_subj)
# get_nrem_epochs()
# sleep_scoring(subjects=all_subj)
plot_before_stim_duration(all_subj)