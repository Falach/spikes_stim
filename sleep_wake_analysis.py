import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import sem, zscore
import glob
from utils import get_stim_starts, edf_path, stim_path


def sleep_wake_per_minute(subjects, norm='baseline'):
    blocks_dict = {'stim_w': [], 'stim_s': [], 'stim_t': [], 'undisturbed_w': [], 'undisturbed_s': [], 'undisturbed_t': []}
    for subj in subjects.keys():
        top_chans = subjects[subj]
        sleep_percent = pd.read_csv(f'results/{subj}_sleep_percent.csv')
        for chan in top_chans:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            baseline = chan_rates['rate'][0]
            # stim blocks, start after row 0 cuase its the baseline
            for i in range(1, len(chan_rates['rate']) - 1, 2):
                y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                          chan_rates['rate_1_20%'][i],
                          chan_rates['rate_2_20%'][i],
                          chan_rates['rate_3_20%'][i],
                          chan_rates['rate_4_20%'][i],
                          chan_rates['rate_5_20%'][i]]
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                elif norm == 'z':
                    y_axis = zscore(y_axis)
                if sleep_percent['sleep_percent'][i] == 100:
                    blocks_dict['stim_s'].append(y_axis)
                elif sleep_percent['sleep_percent'][i] == 0:
                    blocks_dict['stim_w'].append(y_axis)
                else:
                    blocks_dict['stim_t'].append(y_axis)

            # undisturbed blocks, start after row 0 cuase its the baseline
            for i in range(2, len(chan_rates['rate']), 2):
                y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                          chan_rates['rate_1_20%'][i],
                          chan_rates['rate_2_20%'][i],
                          chan_rates['rate_3_20%'][i],
                          chan_rates['rate_4_20%'][i],
                          chan_rates['rate_5_20%'][i]]
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                elif norm == 'z':
                    y_axis = zscore(y_axis)
                if sleep_percent['sleep_percent'][i] == 100:
                    blocks_dict['undisturbed_s'].append(y_axis)
                elif sleep_percent['sleep_percent'][i] == 0:
                    blocks_dict['undisturbed_w'].append(y_axis)
                else:
                    blocks_dict['undisturbed_t'].append(y_axis)

    return blocks_dict


def plot_all_vigilance(subjects, norm='baseline'):
    plt.rcParams['figure.figsize'] = [8, 5]
    blocks_dict = sleep_wake_per_minute(subjects, norm=norm)
    for block_type in blocks_dict.keys():
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            avg_df = pd.DataFrame(blocks_dict[block_type], columns=['0', '1', '2', '3', '4', '5'])
            means = [avg_df[str(i)].mean() for i in range(0, 6)]
            stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, 6)]
            plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o',
                         label=f'{block_type}')
        else:
            plt.errorbar(0, 0)
    plt.legend()
    if norm == 'baseline':
        plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'average block per vigilance')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')
    print(1)


def plot_triple_vigilance(subjects, norm='baseline'):
    blocks_dict = sleep_wake_per_minute(subjects, norm=norm)
    for block_type in [key for key in blocks_dict.keys() if 'stim' in key]:
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            avg_df = pd.DataFrame(blocks_dict[block_type], columns=['0', '1', '2', '3', '4', '5'])
            means = [avg_df[str(i)].mean() for i in range(0, 6)]
            stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, 6)]
            plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o', label=f'{block_type}-{str(blocks_num)} blocks')
        else:
            plt.errorbar(0, 0)
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'average stim block per vigilance')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')

    plt.clf()
    for block_type in [key for key in blocks_dict.keys() if 'stim' not in key]:
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            avg_df = pd.DataFrame(blocks_dict[block_type], columns=['0', '1', '2', '3', '4', '5'])
            means = [avg_df[str(i)].mean() for i in range(0, 6)]
            stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, 6)]
            plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o', label=f'{block_type}-{str(blocks_num)} blocks')
        else:
            plt.errorbar(0, 0)
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'average undisturbed block per vigilance')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')


def plot_double_vigilance(subjects, norm='baseline'):
    blocks_dict = sleep_wake_per_minute(subjects, norm)
    colors = {'stim_w': 'tab:blue', 'stim_s': 'tab:orange', 'stim_t': 'tab:green', 'undisturbed_w': 'tab:red',
              'undisturbed_s': 'tab:purple', 'undisturbed_t': 'tab:brown'}
    for block_type in [key for key in blocks_dict.keys() if '_w' in key]:
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            avg_df = pd.DataFrame(blocks_dict[block_type], columns=['0', '1', '2', '3', '4', '5'])
            means = [avg_df[str(i)].mean() for i in range(0, 6)]
            stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, 6)]
            plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o', color=colors[block_type],
                         label=f'{block_type}-{str(blocks_num)} blocks')
        else:
            plt.errorbar(0, 0)
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'average wake block per stim')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')

    plt.clf()
    for block_type in [key for key in blocks_dict.keys() if '_s' in key]:
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            avg_df = pd.DataFrame(blocks_dict[block_type], columns=['0', '1', '2', '3', '4', '5'])
            means = [avg_df[str(i)].mean() for i in range(0, 6)]
            stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, 6)]
            plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o', color=colors[block_type],
                         label=f'{block_type}-{str(blocks_num)} blocks')
        else:
            plt.errorbar(0, 0)
    plt.legend()
    if norm == 'baseline':
        plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'average sleep block per stim')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')

    plt.clf()
    for block_type in [key for key in blocks_dict.keys() if '_t' in key]:
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            avg_df = pd.DataFrame(blocks_dict[block_type], columns=['0', '1', '2', '3', '4', '5'])
            means = [avg_df[str(i)].mean() for i in range(0, 6)]
            stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, 6)]
            plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o', color=colors[block_type],
                         label=f'{block_type}-{str(blocks_num)} blocks')
        else:
            plt.errorbar(0, 0)
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'average transition block per stim')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')


def sleep_wake_blocks(subjects, norm='baseline'):
    baselines = []
    blocks_dict = {'stim_w': [], 'stim_s': [], 'stim_t': [], 'undisturbed_w': [], 'undisturbed_s': [], 'undisturbed_t': []}
    colors = {'stim_w': 'tab:blue', 'stim_s': 'tab:orange', 'stim_t': 'tab:green', 'undisturbed_w': 'tab:red',
              'undisturbed_s': 'tab:purple', 'undisturbed_t': 'tab:brown'}
    for subj in subjects.keys():
        top_chans = subjects[subj]
        sleep_percent = pd.read_csv(f'results/{subj}_sleep_percent.csv')
        for chan in top_chans:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            baseline = chan_rates['rate'][0]
            baselines.append(baseline)
            # stim blocks, start after row 0 cuase its the baseline
            for i in range(1, len(chan_rates['rate']) - 1, 2):
                y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                          chan_rates['rate'][i]]
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                if sleep_percent['sleep_percent'][i] == 100:
                    blocks_dict['stim_s'].append(y_axis)
                elif sleep_percent['sleep_percent'][i] == 0:
                    blocks_dict['stim_w'].append(y_axis)
                else:
                    blocks_dict['stim_t'].append(y_axis)

            # undisturbed blocks, start after row 0 cuase its the baseline
            for i in range(2, len(chan_rates['rate']), 2):
                y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                          chan_rates['rate'][i]]
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                if sleep_percent['sleep_percent'][i] == 100:
                    blocks_dict['undisturbed_s'].append(y_axis)
                elif sleep_percent['sleep_percent'][i] == 0:
                    blocks_dict['undisturbed_w'].append(y_axis)
                else:
                    blocks_dict['undisturbed_t'].append(y_axis)

    for block_type in blocks_dict.keys():
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            avg_df = pd.DataFrame(blocks_dict[block_type], columns=['baseline', 'block avg'])
            means = [avg_df[str(i)].mean() for i in ['baseline', 'block avg']]
            stds = [sem(avg_df[str(i)], nan_policy='omit') for i in ['baseline', 'block avg']]
            # plt.errorbar(['baseline', 'block avg'], means, yerr=stds, capsize=5, fmt='-o',
            #              label=f'{block_type}-{str(blocks_num)} blocks')
            rect = plt.bar([block_type], means[1], yerr=stds[1], capsize=5, alpha=0.7, color=colors[block_type])
            # plt.bar_label(rect, [f'n={blocks_num}'], padding=1)
            plt.bar_label(rect, [f'avg={round(means[1], 1)}'], padding=1)
            plt.scatter([block_type] * len(avg_df['block avg'].tolist()), avg_df['block avg'].tolist(), color=colors[block_type])
        else:
            rect = plt.bar([block_type], 0)

    for block_type in blocks_dict.keys():
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            plt.text([block_type], plt.ylim()[1] - 10, f'n={blocks_num}', size=10)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'average block per vigilance and stim- avg baseline: {round(np.array(baseline).mean(), 2)}')
    plt.ylabel('% Change in spike rate')
    plt.locator_params(axis='y', nbins=12)
    plt.tight_layout()
    return blocks_dict

def transition_blocks(subjects, norm='baseline', separate=True):
    baselines = []
    blocks_dict = {'stim_w_s': [], 'stim_s_w': [], 'stim_multi': [], 'undisturbed_w_s': [],
                   'undisturbed_s_w': [], 'undisturbed_multi': []}
    colors = {'stim_w_s': 'tab:blue', 'stim_s_w': 'tab:orange', 'stim_multi': 'tab:green', 'undisturbed_w_s': 'tab:red',
              'undisturbed_s_w': 'tab:purple', 'undisturbed_multi': 'tab:brown'}
    for subj in subjects.keys():
        top_chans = subjects[subj]
        sleep_percent = pd.read_csv(f'results/{subj}_sleep_percent.csv')
        for chan in top_chans:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            baseline = chan_rates['rate'][0]
            # stim blocks, start after row 0 cuase its the baseline
            for i in range(1, len(chan_rates['rate']) - 1, 2):
                y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                          chan_rates['rate'][i]]
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                if 0 < sleep_percent['sleep_percent'][i] < 100:
                    if sleep_percent['transition_type'][i] == 'w-s':
                        blocks_dict['stim_w_s'].append(y_axis)
                    elif sleep_percent['transition_type'][i] == 's-w':
                        blocks_dict['stim_s_w'].append(y_axis)
                    else:
                        blocks_dict['stim_multi'].append(y_axis)

            # undisturbed blocks, start after row 0 cuase its the baseline
            for i in range(2, len(chan_rates['rate']), 2):
                y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                          chan_rates['rate'][i]]
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                if 0 < sleep_percent['sleep_percent'][i] < 100:
                    if sleep_percent['transition_type'][i] == 'w-s':
                        blocks_dict['undisturbed_w_s'].append(y_axis)
                    elif sleep_percent['transition_type'][i] == 's-w':
                        blocks_dict['undisturbed_s_w'].append(y_axis)
                    else:
                        blocks_dict['undisturbed_multi'].append(y_axis)
    if not separate:
        blocks_dict = {'w_s': blocks_dict['stim_w_s'] + blocks_dict['undisturbed_w_s'],
                       's_w': blocks_dict['stim_s_w'] + blocks_dict['undisturbed_s_w'],
                       'multi': blocks_dict['stim_multi'] + blocks_dict['undisturbed_multi']}
        colors = {'w_s': 'tab:blue', 's_w': 'tab:orange', 'multi': 'tab:green'}
    for block_type in blocks_dict.keys():
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            avg_df = pd.DataFrame(blocks_dict[block_type], columns=['baseline', 'block avg'])
            means = [avg_df[str(i)].mean() for i in ['baseline', 'block avg']]
            stds = [sem(avg_df[str(i)], nan_policy='omit') for i in ['baseline', 'block avg']]
            # plt.errorbar(['baseline', 'block avg'], means, yerr=stds, capsize=5, fmt='-o',
            #              label=f'{block_type}-{str(blocks_num)} blocks')
            rect = plt.bar([block_type], means[1], yerr=stds[1], capsize=5, alpha=0.7, color=colors[block_type])
            # plt.bar_label(rect, [f'n={blocks_num}'], padding=1)
            plt.bar_label(rect, [f'avg={round(means[1], 1)}'], padding=1)
            plt.scatter([block_type] * len(avg_df['block avg'].tolist()), avg_df['block avg'].tolist(), color=colors[block_type])
        else:
            rect = plt.bar([block_type], 0)

    for block_type in blocks_dict.keys():
        blocks_num = len(blocks_dict[block_type])
        if blocks_num > 0:
            plt.text([block_type], plt.ylim()[1] - 10, f'n={blocks_num}', size=10)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title(f'average block per vigilance and stim')
    plt.ylabel('% Change in spike rate')
    plt.locator_params(axis='y', nbins=12)
    plt.tight_layout()
    return blocks_dict

manual_select_14_thresh = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '487': ['LAH2', 'LA2', 'LEC1'],
                           '489': ['LPHG2', 'RAH1', 'RPHG1'], '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'],
                           '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'], '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'],
                           '510-7': ['RAH1'], '515': ['RA1', 'RAH2'], '520': ['REC1', 'RMH1', 'LMH1'],
                           '544': ['LEC1', 'LOPR-MI1'], '545': ['LAH3', 'REC1']}
wake_baseline = ['497', '515', '545']
transition_baseline = ['498']

# plot results of all 10
# sleep_wake_blocks(manual_select_14_thresh)
# plot_all_vigilance(manual_select_14_thresh)
# plot_triple_vigilance(manual_select_14_thresh)
# plot_double_vigilance(manual_select_14_thresh)

# plot only sleep baseline
# curr_subjects = {key: value for key, value in manual_select_14_thresh.items() if key not in wake_baseline + transition_baseline}
# sleep_wake_blocks(curr_subjects)
# plot_all_vigilance(curr_subjects)
# plot_triple_vigilance(curr_subjects)
# plot_double_vigilance(curr_subjects)

# plot only wake baseline
# curr_subjects = {key: value for key, value in manual_select_14_thresh.items() if key  in wake_baseline}
# sleep_wake_blocks(curr_subjects)
# plot_all_vigilance(curr_subjects)
# plot_triple_vigilance(curr_subjects)
# plot_double_vigilance(curr_subjects)


# again with 487 and 544 and minmax baseline
# sleep_wake_blocks(manual_select_14_thresh, norm='minmax') # no meaning for norm because of only 2 points
# plot_all_vigilance(manual_select_14_thresh, norm='z')
# plot_triple_vigilance(manual_select_14_thresh, norm='z')
# plot_double_vigilance(manual_select_14_thresh, norm='z')


# ipsi
stim_side_right = ['485', '487', '489', '490', '496', '497', '510-1', '510-7', '515', '520', '538', '544', '545']
manual_ipsi = {}
for subj in manual_select_14_thresh.keys():
    curr_side = 'R' if subj in stim_side_right else 'L'
    top_chans = [x for x in manual_select_14_thresh[subj] if x[0] == curr_side]
    if len(top_chans) > 0:
        manual_ipsi[subj] = top_chans
#
# sleep_wake_blocks(manual_ipsi)
# curr_subjects = {key: value for key, value in manual_ipsi.items() if key not in wake_baseline + transition_baseline}
# sleep_wake_blocks(curr_subjects)
# curr_subjects = {key: value for key, value in manual_ipsi.items() if key  in wake_baseline}
# sleep_wake_blocks(curr_subjects)

# transitions
# transition_blocks(manual_select_14_thresh)
# transition_blocks(manual_ipsi)
transition_blocks(manual_select_14_thresh, separate=False)
transition_blocks(manual_ipsi, separate=False)


# plot_per_min_transition(curr_subjects)