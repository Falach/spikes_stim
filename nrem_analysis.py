import mne
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob
from utils import get_stim_starts, edf_path, stim_path
from scipy.stats import sem


def chans_nrem_profile(subjects):
    plt.rcParams['figure.figsize'] = [8, 5]
    x_axis = []
    for subj in subjects.keys():
        top_chans = subjects[subj]
        all_chans = []
        for chan in top_chans:
            chan_rates = pd.read_csv('results\\%s_%s_nrem_rates.csv' % (subj, chan), index_col=0)
            nrem_rates = chan_rates[chan_rates.index % 2 == 1].reset_index(drop=True)
            during_stim = chan_rates[chan_rates.is_stim == 1]['rate'].tolist()
            stim_start = nrem_rates.loc[nrem_rates.is_stim == 1].index[0]
            if stim_start == 1:
                before_stim = []
            else:
                before_stim = nrem_rates.iloc[:stim_start]['rate'].tolist()
            stim_end = nrem_rates.loc[nrem_rates.is_stim == 1].index[-1]
            if stim_end == len(nrem_rates) - 1:
                after_stim = []
            else:
                after_stim = nrem_rates.iloc[stim_end + 1:]['rate'].tolist()
            y_axis = chan_rates['rate'].tolist()
            # all_chans.append(y_axis)
            plt.plot(range(-1, len(nrem_rates['rate'].tolist())),
                     [chan_rates.iloc[0]['rate']] + nrem_rates['rate'].tolist(), '-o', label=chan)

        # mark_start = 0 if stim_start[0] == 0 else stim_start[0] - 1
        # mark_end = stim_end + 1 if len(stim_start) != 1 else mark_start + 1
        plt.axvspan(stim_start - 1, stim_end if stim_start != stim_end else stim_start, facecolor='silver', alpha=0.5)
        plt.xlabel('NREM epoch')
        plt.ylabel('spike rate')
        plt.title(f'{subj} channels NREM profile')
        plt.xticks(list(range(-1, len(nrem_rates['rate']))), ['wake/REM'] + list(range(0, len(nrem_rates['rate']))))
        # plt.xticks(['wake/REM'] + list(range(0, len(nrem_rates['rate']))))
        plt.legend()
        plt.savefig(f'results/{subj}_nrem_rates')
        plt.clf()

def before_during_after_stim_nrem(subjects, norm='base'):
    plt.rcParams['figure.figsize'] = [8, 5]
    stim_val = {'before': 0, 'during': 1, 'after': 2}
    all_means = []
    for subj in subjects.keys():
        top_chans = subjects[subj]
        all_chans = []
        for chan in top_chans:
            curr_chan = []
            chan_rates = pd.read_csv('results\\%s_%s_nrem_split_rates.csv' % (subj, chan), index_col=0)
            for i in stim_val.keys():
                curr_rates = chan_rates[chan_rates.is_stim == stim_val[i]]
                total_duration = curr_rates['duration_sec'].sum() / 60
                total_spikes = curr_rates['n_spikes'].sum()
                if total_duration == 0:
                    curr_chan.append(-1)
                else:
                    curr_chan.append(total_spikes / total_duration)
            # normalize by max
            if norm == 'max':
                norm_chan = [x / max(curr_chan) for x in curr_chan]
            elif norm == 'mean':
                norm_chan = [x / np.mean(curr_chan) for x in curr_chan]
            elif norm == 'base':
                norm_chan = [(x - curr_chan[0])/ max(x, curr_chan[0]) for x in curr_chan if x is not None]
            else:
                norm_chan = curr_chan
            all_chans.append(norm_chan)
        mean_rates = np.mean(all_chans, axis=0)
        plt.plot(list(stim_val.keys()), mean_rates, '-o', label=chan)
        all_means.append(mean_rates)
    rect = plt.bar(stim_val.keys(), np.mean(all_means, axis=0), alpha=0.5)
    plt.bar_label(rect, padding=1)
    plt.axhline(y=0, color='black', linestyle='dashed', label='After')
    plt.ylabel('spike rate')
    plt.title(f'channels NREM profile')
    # plt.xticks([0, 1, 2], ['before', 'during', 'after'])
    plt.savefig(f'results/before_after_nrem_rates_{norm}')

def compare_two_stim(subjects, stim_val={'during': 1, 'after': 2}, norm=False):
    plt.rcParams['figure.figsize'] = [8, 5]
    all_means = []
    for subj in subjects.keys():
        top_chans = subjects[subj]
        all_chans = []
        for chan in top_chans:
            curr_chan = []
            chan_rates = pd.read_csv('results\\%s_%s_nrem_split_rates.csv' % (subj, chan), index_col=0)
            for i in stim_val.keys():
                curr_rates = chan_rates[chan_rates.is_stim == stim_val[i]]
                total_duration = curr_rates['duration_sec'].sum() / 60
                total_spikes = curr_rates['n_spikes'].sum()
                if total_duration == 0:
                    curr_chan.append(-1)
                else:
                    curr_chan.append(total_spikes / total_duration)
            # normalize by max
            if norm:
                norm_chan = [0] + [(curr_chan[1] - curr_chan[0])/ max(curr_chan)]
            else:
                norm_chan = curr_chan
            all_chans.append(norm_chan)
        mean_rates = np.mean(all_chans, axis=0)
        plt.plot(list(stim_val.keys()), mean_rates, '-o', label=chan)
        all_means.append(mean_rates[1])
    rect = plt.bar(stim_val.keys(), [0, np.mean(all_means)], alpha=0.5)
    plt.bar_label(rect, padding=1)
    plt.axhline(y=0, color='black', linestyle='dashed', label='After')
    plt.ylabel('spike rate')
    plt.title(f'channels NREM profile')
    # plt.xticks([0, 1, 2], ['before', 'during', 'after'])
    plt.savefig(f'results/before_after_nrem_rates_{norm}')

def compare_hemisphere(subjects, norm=False):
    print()


frontal_stim = ['485', '486', '487', '488', '497', '498', '499', '505', '510-1', '510-7', '515', '520', '545']
# temporal_stim = ['489', '490', '496', '538']
mixed_stim = ['485', '497', '499', '505', '510-1', '510-7']
all_except_490_520 = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '538', '541', '544', '545']
manual_select = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '487': ['LAH2', 'LA2', 'LEC1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'], '515': ['RA1', 'RAH2'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}

manual_lateral = {'485': ['RPHG6'], '486': ['RPSMA5'], '488': ['LA7'],
                 '497': ['REC6', 'RPHG5', 'LAH6', 'LA5', 'LPHG5', 'RMH7'],
                 '499': ['LMH5'], '505': ['LEC5']}
manual_mesial = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '487': ['LAH2', 'LA2', 'LEC1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG1'],
                 '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'], '515': ['RA1', 'RAH2'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
manual_select_no_515_545 = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '487': ['LAH2', 'LA2', 'LEC1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'],
                 '520': ['REC1', 'RMH1', 'LMH1']}

control_chans = {'396': ['LAH1', 'LPHG2', 'LMH1', 'LEC1', 'LA2'], '398': ['LA1', 'LAH1', 'RAH1', 'RA2', 'REC3', 'LPHG1'],
                 '402': ['LA1', 'RAH1', 'LAH1', 'RA2', 'REC1'], '406': ['LPHG2', 'RPHG1', 'RAH1', 'RA2', 'LAH1', 'RMPF3'],
                 '415': ['LAH1', 'RA1', 'RAH1', 'LEC1', 'REC1'], '416': ['LAH2', 'RA2', 'RAH1', 'RIFGA1']}

control_chans_no_415 = {'396': ['LAH1', 'LPHG2', 'LMH1', 'LEC1', 'LA2'], '398': ['LA1', 'LAH1', 'RAH1', 'RA2', 'REC3', 'LPHG1'],
                 '402': ['LA1', 'RAH1', 'LAH1', 'RA2', 'REC1'], '406': ['LPHG2', 'RPHG1', 'RAH1', 'RA2', 'LAH1', 'RMPF3'],
                 '416': ['LAH2', 'RA2', 'RAH1', 'RIFGA1']}

# chans_nrem_profile(manual_select)
# TODO: add 515 and 545
# before_during_after_stim_nrem(control_chans_no_415)
before_during_after_stim_nrem(manual_select_no_515_545)



# before_during_after_stim_nrem(control_chans)
