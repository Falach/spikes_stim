import mne
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob


def calc_rate_per_chan(subjects):
    for subj in subjects:
        subj_files_list = glob.glob(f'C:\\repos\\epileptic_activity\\results\\{subj}*rates*')
        rates_per_chan = {'channel': [], 'baseline': [], 'sum': []}
        for i, curr_file in enumerate(subj_files_list):
            ch_name = curr_file.split(f'{subj}_')[1].split('_rates')[0]
            chan_rates = pd.read_csv(curr_file, index_col=0)
            rates_per_chan['channel'].append(ch_name)
            rates_per_chan['baseline'].append(chan_rates['rate'][0])
            rates_per_chan['sum'].append(chan_rates['n_spikes'].sum())

        df = pd.DataFrame(rates_per_chan)
        df.to_csv(f'C:\\repos\\epileptic_activity\\results\\{subj}_chan_sum.csv')


def plot_first_block(subj, channels, file_name):
    blocks = []
    for chan in channels:
        chan_rates = pd.read_csv(f'C:\\repos\\epileptic_activity\\results\\{subj}_{chan}_rates.csv', index_col=0)
        y_axis = [chan_rates['rate'][0],  # the previous stop baseline
                  chan_rates['rate_1_20%'][1],
                  chan_rates['rate_2_20%'][1],
                  chan_rates['rate_3_20%'][1],
                  chan_rates['rate_4_20%'][1],
                  chan_rates['rate_5_20%'][1]]
        blocks.append(y_axis)

    avg_df = pd.DataFrame(blocks, columns=['0', '1', '2', '3', '4', '5'])
    means = [avg_df[str(i)].mean() for i in range(0, 6)]
    stds = [avg_df[str(i)].std() for i in range(0, 6)]
    plt.errorbar(list(range(0, 6)), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
    plt.title(f'{subj} - {file_name}')
    plt.xlabel('Time point')
    plt.ylabel('Spikes per minute')
    plt.savefig(f'C:\\repos\\epileptic_activity\\results\\{subj}_{file_name}.png')
    plt.clf()


def plot_all_subjects_top_10(subjects, file_name):
    all_avg = []
    for subj in subjects:
        df = pd.read_csv(f'C:\\repos\\epileptic_activity\\results\\{subj}_chan_sum.csv')
        # Top 10
        channels = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()
        blocks = []
        for chan in channels:
            chan_rates = pd.read_csv(f'C:\\repos\\epileptic_activity\\results\\{subj}_{chan}_rates.csv', index_col=0)
            y_axis = [chan_rates['rate_5_20%'][0],  # the previous stop baseline
                      chan_rates['rate_1_20%'][1],
                      # chan_rates['rate_2_20%'][1],
                      # chan_rates['rate_3_20%'][1],
                      # chan_rates['rate_4_20%'][1],
                      chan_rates['rate_5_20%'][1]]
            blocks.append(y_axis)

        avg_df = pd.DataFrame(blocks, columns=['baseline', '1 min', '5 min'])
        means = [avg_df[i].mean() for i in ['baseline', '1 min', '5 min']]
        all_avg.append(means)
        stds = [avg_df[i].std() for i in ['baseline', '1 min', '5 min']]
        plt.plot(['baseline', '1 min', '5 min'], means, 'o-')

    # plt.plot()
    plt.title(f'all subjects- {file_name}')
    plt.xlabel('Time point')
    plt.ylabel('Spikes per minute')
    # plt.show()
    plt.savefig(f'C:\\repos\\epileptic_activity\\results\\all_{file_name}.png')
    plt.clf()


def plot_all_subjects_frontal(subjects, file_name):
    all_avg = []
    for subj in subjects:
        df = pd.read_csv(f'C:\\repos\\epileptic_activity\\results\\{subj}_chan_sum.csv')
        # get frontal channels
        channels = df[df['channel'].str.contains('F')]['channel'].tolist()
        blocks = []
        for chan in channels:
            chan_rates = pd.read_csv(f'C:\\repos\\epileptic_activity\\results\\{subj}_{chan}_rates.csv', index_col=0)
            y_axis = [chan_rates['rate_5_20%'][0],  # the previous stop baseline
                      chan_rates['rate_1_20%'][1],
                      # chan_rates['rate_2_20%'][1],
                      # chan_rates['rate_3_20%'][1],
                      # chan_rates['rate_4_20%'][1],
                      chan_rates['rate_5_20%'][1]]
            blocks.append(y_axis)

        avg_df = pd.DataFrame(blocks, columns=['baseline', '1 min', '5 min'])
        means = [avg_df[i].mean() for i in ['baseline', '1 min', '5 min']]
        all_avg.append(means)
        stds = [avg_df[i].std() for i in ['baseline', '1 min', '5 min']]
        plt.plot(['baseline', '1 min', '5 min'], means, 'o-')

    # plt.plot()
    plt.title(f'all subjects- {file_name}')
    plt.xlabel('Time point')
    plt.ylabel('Spikes per minute')
    # plt.show()
    plt.savefig(f'C:\\repos\\epileptic_activity\\results\\all_{file_name}.png')
    plt.clf()

# calc rate per chan and plot top ten
# all = ['485', '486', '487', '488', '489', '490', '496', '497', '498', '499', '505', '510', '515', '520', '538',
#        '541', '544', '545']
subjects = ['485', '486', '487', '488', '489', '496', '497', '498', '505', '515', '538', '541', '544', '545']
frontal_stim = ['485', '486', '487', '488', '497', '498', '505', '515', '541', '544', '545']
temporal_stim = ['489', '490', '496', '538']
# calc_rate_per_chan(subjects)
# for subj in subjects:
#     df = pd.read_csv(f'C:\\repos\\epileptic_activity\\results\\{subj}_chan_sum.csv')
#     top_10 = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()
#     plot_first_block(subj, top_10, 'top_10')

for subj in frontal_stim:
    df = pd.read_csv(f'C:\\repos\\epileptic_activity\\results\\{subj}_chan_sum.csv')
    frontal_chans = df[df['channel'].str.contains('F')]['channel'].tolist()
    plot_first_block(subj, frontal_chans, 'frontal')

# plot_all_subjects_top_10(subjects, 'top_10_block_1')
plot_all_subjects_frontal(frontal_stim, 'frontal_channels_block_1')