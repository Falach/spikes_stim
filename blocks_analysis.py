import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob
from utils import get_stim_starts, edf_path, stim_path

# TODO: not informative cause the baseline is normalized and the avg ignore trend
def plot_subj_undisturbed(subjects, avg=True, norm='baseline'):
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        top_chan = df.sort_values(by='baseline', ascending=False).iloc[:1, :]['channel'].tolist()[0]
        # TODO: filter is_stim = 0
        chan_rates = pd.read_csv(f'results\\{subj}_{top_chan}_rates.csv', index_col=0)
        # draw the baseline before stim and after all stims
        plt.axhline(y=chan_rates['rate'][0], color='b', linestyle='dashed', label='Before')
        plt.axhline(y=chan_rates['rate'][len(chan_rates['rate']) - 1], color='r', linestyle='dashed', label='After')
        for_avg = []
        # draw each stop as 5 rates
        for i in range(0, len(chan_rates['rate']) - 2, 2):
            y_axis = [chan_rates['rate_5_20%'][i],  # the previous stop baseline
                      chan_rates['rate_1_20%'][i + 2],
                      chan_rates['rate_2_20%'][i + 2],
                      chan_rates['rate_3_20%'][i + 2],
                      chan_rates['rate_4_20%'][i + 2],
                      chan_rates['rate_5_20%'][i + 2]]
            if norm == 'baseline':
                # TODO: how to norm when baseline is 0
                baseline = y_axis[0] if y_axis[0] > 0 else 1
                y_axis = np.array(y_axis) / baseline
                y_axis = y_axis * 100 - 100
            for_avg.append(y_axis)
            if not avg:
                plt.plot(list(range(0, 6)), y_axis, 'o-', label=str(i + 1))

        # plot avg trend
        avg_df = pd.DataFrame(for_avg, columns=['0', '1', '2', '3', '4', '5'])
        means = [avg_df[str(i)].mean() for i in range(0, 6)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, 6)]
        plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        plt.legend()
        plt.title(f'{subj} - {top_chan} undisturbed blocks')
        plt.xlabel('Time point')
        plt.ylabel('Spikes per minute')
        filename = f'results/{subj}_{top_chan}_undisturbed_blocks'
        if avg:
            filename += '_avg'
        if norm:
            filename += '_norm'
        plt.savefig(filename)
        plt.clf()


def plot_first_block(subj, channels, file_name):
    blocks = []
    for chan in channels:
        chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
        y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                  chan_rates['rate_1_20%'][1],
                  chan_rates['rate_2_20%'][1],
                  chan_rates['rate_3_20%'][1],
                  chan_rates['rate_4_20%'][1],
                  chan_rates['rate_5_20%'][1]]
        blocks.append(y_axis)

    avg_df = pd.DataFrame(blocks, columns=['0', '1', '2', '3', '4', '5'])
    means = [avg_df[str(i)].mean() for i in range(0, 6)]
    stds = [avg_df[str(i)].std(ddof=0) for i in range(0, 6)]
    plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
    plt.title(f'{subj} - {file_name}')
    plt.xlabel('Time point')
    plt.ylabel('Spikes per minute')
    plt.savefig(f'results\\{subj}_{file_name}.png')
    plt.clf()


def plot_all_subjects_top_10(subjects, chan_num=10, norm=False):
    all_avg = []
    counter1, counter2 = 0, 0
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        channels = df.sort_values(by='baseline', ascending=False).iloc[:chan_num, :]['channel'].tolist()
        blocks = []
        for chan in channels:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            y_axis = [chan_rates['rate'][0],  # the baseline
                      chan_rates['rate_1_20%'][2],
                      chan_rates['rate_5_20%'][2]]
            if norm:
                # TODO: how to norm when baseline is 0
                y_axis[0] = y_axis[0] if y_axis[0] > 0 else 1
                y_axis = np.array(y_axis) / y_axis[0]
                y_axis = y_axis * 100 - 100
            blocks.append(y_axis)

        avg_df = pd.DataFrame(blocks, columns=['baseline', '1 min', '5 min'])
        means = [avg_df[i].mean() for i in ['baseline', '1 min', '5 min']]
        all_avg.append(means)
        stds = [avg_df[i].std(ddof=0) for i in ['baseline', '1 min', '5 min']]
        plt.scatter(['baseline', '1 min', '5 min'], means)
        # first line
        if means[1] - means[0] > 0:
            color = 'g'
            counter1 += 1
        else:
            color = 'r'
        plt.plot(['baseline', '1 min'], means[:2], c=color)
        # second line
        if means[2] - means[1] > 0:
            color = 'g'
            counter2 += 1
        else:
            color = 'r'
        plt.plot(['1 min', '5 min'], means[1:], c=color)

    all_avg_df = pd.DataFrame(all_avg, columns=['baseline', '1 min', '5 min'])
    all_means = [all_avg_df[i].mean() for i in ['baseline', '1 min', '5 min']]
    plt.bar(['baseline', '1 min', '5 min'], all_means, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'first undisturbed block- top {chan_num} channels, {len(subjects)} subj')
    plt.xlabel('Time point')
    plt.ylabel('Spikes per minute')
    plt.text(0, 200, f'increase: {counter1}')
    plt.text(1, 200, f'increase: {counter2}')
    plt.savefig(f'results\\all_first_undisturbed_block_top{chan_num}.png')


def plot_all_subjects_first_block(subjects, norm=True, colors='auto', stim=False):
    all_avg = []
    for subj in subjects.keys():
        channels = subjects[subj]
        blocks = []
        for chan in channels:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                      chan_rates['rate_1_20%'][1 if stim else 2],
                      chan_rates['rate_5_20%'][1 if stim else 2]]
            if norm:
                y_axis[0] = y_axis[0] if y_axis[0] > 0 else 1
                y_axis = np.array(y_axis) / y_axis[0]
                y_axis = y_axis * 100 - 100
            blocks.append(y_axis)

        avg_df = pd.DataFrame(blocks, columns=['baseline', '1 min', '5 min'])
        means = [avg_df[i].mean() for i in ['baseline', '1 min', '5 min']]
        all_avg.append(means)
        if colors == 'auto':
            plt.plot(['baseline', '1 min', '5 min'], means, '-o')
        else:
            plt.scatter(['baseline', '1 min', '5 min'], means)
            # first line
            if means[1] - means[0] > 0:
                color = 'g'
            else:
                color = 'r'
            plt.plot(['baseline', '1 min'], means[:2], c=color)
            # second line
            if means[2] - means[1] > 0:
                color = 'g'
            else:
                color = 'r'
            plt.plot(['1 min', '5 min'], means[1:], c=color)

    plt.axhline(y=0, color='black', linestyle='--')
    counter1 = len([x for x in all_avg if x[1] - x[0] > 0])
    counter2 = len([x for x in all_avg if x[2] - x[1] > 0])
    all_avg_df = pd.DataFrame(all_avg, columns=['baseline', '1 min', '5 min'])
    all_means = [all_avg_df[i].mean() for i in ['baseline', '1 min', '5 min']]
    plt.bar(['baseline', '1 min', '5 min'], all_means, alpha=0.5)
    plt.title(f'First undisturbed block- top channels avg, {len(subjects)} subj')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')
    plt.text(0, plt.ylim()[1] - 10, f'increase: {counter1}')
    plt.text(1, plt.ylim()[1] - 10, f'increase: {counter2}')
    plt.savefig(f'results\\first_undisturbed_block_top_chans.png')


def plot_all_subjects_all_blocks(subjects, norm=True, colors='auto', stim=False):
    all_avg = []
    for subj in subjects.keys():
        channels = subjects[subj]
        blocks = []
        for chan in channels:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            block_range = range(1, len(chan_rates), 2) if stim else range(2, len(chan_rates), 2)
            for i in block_range:
                y_axis = [chan_rates['rate'][0],
                          chan_rates['rate_1_20%'][i],
                          chan_rates['rate_5_20%'][i]]
                if norm:
                    y_axis[0] = y_axis[0] if y_axis[0] > 0 else 1
                    y_axis = np.array(y_axis) / y_axis[0]
                    y_axis = y_axis * 100 - 100
                blocks.append(y_axis)

        avg_df = pd.DataFrame(blocks, columns=['baseline', '1 min', '5 min'])
        means = [avg_df[i].mean() for i in ['baseline', '1 min', '5 min']]
        all_avg.append(means)
        if colors == 'auto':
            plt.plot(['baseline', '1 min', '5 min'], means, '-o')
        else:
            plt.scatter(['baseline', '1 min', '5 min'], means)
            # first line
            if means[1] - means[0] > 0:
                color = 'g'
            else:
                color = 'r'
            plt.plot(['baseline', '1 min'], means[:2], c=color)
            # second line
            if means[2] - means[1] > 0:
                color = 'g'
            else:
                color = 'r'
            plt.plot(['1 min', '5 min'], means[1:], c=color)

    plt.axhline(y=0, color='black', linestyle='--')
    counter1 = len([x for x in all_avg if x[1] - x[0] > 0])
    counter2 = len([x for x in all_avg if x[2] - x[1] > 0])
    all_avg_df = pd.DataFrame(all_avg, columns=['baseline', '1 min', '5 min'])
    all_means = [all_avg_df[i].mean() for i in ['baseline', '1 min', '5 min']]
    plt.bar(['baseline', '1 min', '5 min'], all_means, alpha=0.5)
    plt.title(f'Average block- top channels avg, {len(subjects)} subj')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')
    plt.text(0, plt.ylim()[1] - 10, f'increase: {counter1}')
    plt.text(1, plt.ylim()[1] - 10, f'increase: {counter2}')


all_after_clean = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '520', '538', '545']
manual_select = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '489': ['LPHG2', 'RAH1', 'RPHG1'], '496': ['RMH1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'], '515': ['RA1', 'RAH2'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
manual_select_14_thresh = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '487': ['LAH2', 'LA2', 'LEC1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'], '515': ['RA1', 'RAH2'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
#plot_subj_undisturbed(all_after_clean)
# plot_all_subjects_top_10(['485', '486', '489', '496', '497', '498', '505', '515', '520'], norm=True, chan_num=1)
# plot_all_subjects_first_block(manual_select_14_thresh, norm=True, colors='auto')
# plot_all_subjects_all_blocks(manual_select_14_thresh, norm=True, colors='auto')
# plot_all_subjects_first_block(manual_select_14_thresh, norm=True, colors='auto', stim=True)
# plot_all_subjects_all_blocks(manual_select_14_thresh, norm=True, colors='auto', stim=True)

stim_side_right = ['485', '487', '489', '490', '496', '497', '510-1', '510-7', '515', '520', '538', '544', '545']
# only ipsi hemisphere
manual_ipsi = {}
for subj in manual_select.keys():
    curr_side = 'R' if subj in stim_side_right else 'L'
    top_chans = [x for x in manual_select[subj] if x[0] == curr_side]
    if len(top_chans) > 0:
        manual_ipsi[subj] = top_chans

# plot_all_subjects_first_block(manual_ipsi, norm=True, colors='auto')
# plot_all_subjects_all_blocks(manual_ipsi, norm=True, colors='auto')
# plot_all_subjects_first_block(manual_ipsi, norm=True, colors='auto', stim=True)
plot_all_subjects_all_blocks(manual_ipsi, norm=True, colors='auto', stim=True)