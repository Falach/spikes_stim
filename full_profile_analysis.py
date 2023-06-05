import mne
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob
from utils import get_stim_starts, edf_path, stim_path
from scipy.stats import sem


# sum up all the files per channel to one csv with the baseline and total spikes per channel
def calc_rate_per_chan(subjects, control=False):
    for subj in subjects:
        subj_files_list = glob.glob(f'results\\{subj}*rates*' if not control else f'results\\control\\{subj}*rates*')
        rates_per_chan = {'channel': [], 'baseline': [], 'sum': [], 'duration': []}
        for i, curr_file in enumerate(subj_files_list):
            if 'stim' not in curr_file:
                ch_name = curr_file.split(f'{subj}_')[1].split('_rates')[0]
                chan_rates = pd.read_csv(curr_file, index_col=0)
                rates_per_chan['channel'].append(ch_name)
                rates_per_chan['baseline'].append(chan_rates['rate'][0])
                rates_per_chan['sum'].append(int(chan_rates['n_spikes'].sum()))
                rates_per_chan['duration'].append(int(chan_rates['duration_sec'].sum()))
                # TODO: calc!
                #rates_per_chan['mean_rate'].append(0)

        df = pd.DataFrame(rates_per_chan)
        df.to_csv(f'results\\{subj}_chan_sum.csv')


def top_channel_full_profile(subjects=['485'], norm='minmax'):
    all_subj = []
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        top_chan = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()[0]
        chan_rates = pd.read_csv(f'results\\{subj}_{top_chan}_rates.csv', index_col=0)
        x_axis = list(range(0, len(chan_rates)))
        y_axis = chan_rates['rate'].tolist()
        if norm:
            y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
        all_subj.append(y_axis)
        plt.plot(x_axis, y_axis, '-o')
        for i in range(0, len(chan_rates) - 1, 2):
            plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
        plt.title(f'{subj} - {top_chan} full 5 min profile')
        plt.xlabel('Time point')
        plt.ylabel('Spikes per minute')
        plt.xlim(0, len(chan_rates) - 1)
        plt.xticks(x_axis)
        # plt.show()
        if norm is not None:
            plt.savefig(f'results\\{subj}_top_chan_{top_chan}_5_min_profile_{norm}_norm.png')
        else:
            plt.savefig(f'results\\{subj}_top_chan_{top_chan}_5_min_profile.png')

        plt.clf()

    return all_subj

# TODO: remove?
def top_chan_profile_avg(subjects, type=None):
    all_subj = []
    x_axis = []
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        top_chan = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()[0]
        chan_rates = pd.read_csv(f'results\\{subj}_{top_chan}_rates.csv', index_col=0)
        if len(chan_rates) > len(x_axis):
            x_axis = list(range(0, len(chan_rates)))
        y_axis = chan_rates['rate'].tolist()
        y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
        all_subj.append(y_axis)

    avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, len(x_axis))])
    subj_num = avg_df.count().tolist()
    if type == 'even':
        means = [avg_df[str(i)].mean() for i in range(0, len(x_axis), 2)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis), 2)]
        plt.errorbar(list(range(0, len(means))), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        plt.title(f'top channel avg only breaks')
    elif type == 'odd':
        means = [avg_df[str(i)].mean() for i in [0] + list(range(1, len(x_axis), 2))]
        stds = [avg_df[str(i)].std(ddof=0) for i in [0] + list(range(1, len(x_axis), 2))]
        plt.errorbar(list(range(0, len(means))), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        plt.title(f'top channel avg only stim')
    else:
        means = [avg_df[str(i)].mean() for i in range(0, len(x_axis))]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis))]
        plt.errorbar(x_axis, means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        for i in range(0, len(x_axis) - 1, 2):
            plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
        plt.title(f'top channel avg full profile')

    plt.xlabel('Time point')
    plt.ylabel('Spikes per minute')
    plt.xlim(-0.2, len(means) - 1)
    plt.xticks(list(range(0, len(means))))
    plt.show()
    # plt.savefig(f'results\\{subj}_{file_name}.png')
    # plt.clf()


def top10_chan_profile_avg(subjects, type=None, norm='baseline', chan_num=10, plot_subj=False):
    all_subj = []
    x_axis = []
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        top_chans = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()[:chan_num]
        all_chans = []
        for chan in top_chans:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            if len(chan_rates) > len(x_axis):
                x_axis = list(range(0, len(chan_rates)))
            y_axis = chan_rates['rate'].tolist()
            if norm == 'baseline':
                y_axis = np.array(y_axis) / y_axis[0]
                y_axis = y_axis * 100 - 100
            elif norm == 'minmax':
                y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
            all_chans.append(y_axis)
        avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(chan_rates))])
        means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(chan_rates))]
        std_subj = [avg_df_subj[str(i)].std(ddof=0) for i in range(0, len(chan_rates))]
        all_subj.append(means_subj)
        if plot_subj:
            # save_channels_pic(top_chans, f'results/{subj}_top_channels')
            chans_text = ', '.join(top_chans)
            plt.gcf().text(0.15, 0.01, chans_text, fontsize=10)
            plt.errorbar(range(len(y_axis)), means_subj, yerr=std_subj, capsize=5, fmt='-o', label='avg', color='black')
            for i in range(0, len(x_axis) - 1, 2):
                plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
            plt.title(f'{subj} top {chan_num} channels avg full profile')
            plt.axhline(y=0, color='r', linestyle='--')
            # plt.xlabel('Time point')
            plt.ylabel('% change in spike rate')
            plt.xlim(-0.2, len(y_axis) - 1)
            plt.xticks(list(range(0, len(means_subj))))
            plt.locator_params(axis='y', nbins=15)
            plt.savefig(f'results/{subj}_top{chan_num}_chan_full_profile_avg_5min_{norm}_norm')
            plt.clf()

    avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, len(x_axis))])
    subj_num = avg_df.count().tolist()
    if type == 'even':
        means = [avg_df[str(i)].mean() for i in range(0, len(x_axis), 2)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis), 2)]
        plt.errorbar(list(range(0, len(means))), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        plt.title(f'top {chan_num} channels avg only breaks')
    elif type == 'odd':
        means = [avg_df[str(i)].mean() for i in [0] + list(range(1, len(x_axis), 2))]
        stds = [avg_df[str(i)].std(ddof=0) for i in [0] + list(range(1, len(x_axis), 2))]
        plt.errorbar(list(range(0, len(means))), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        plt.title(f'top {chan_num} channels avg only stim')
    else:
        means = [avg_df[str(i)].mean() for i in range(0, len(x_axis))]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis))]
        plt.errorbar(x_axis, means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        for i in range(0, len(x_axis) - 1, 2):
            plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
        plt.title(f'top {chan_num} channels avg full profile')

    for k, j in enumerate(avg_df.count().tolist()[1:]):
        plt.annotate(str(j), xy=(x_axis[k], 150))
    if norm == 'baseline':
        plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.xlim(-0.2, len(means) - 1)
    # plt.ylim(0, 1)
    # plt.xticks(list(range(0, len(means))))
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/top{chan_num}_chan_full_profile_avg_5min_{norm}_norm')
    plt.clf()
    # plt.show()


def top_chan_minute_profile_avg(subjects, type=None, norm='baseline', chan_num=1):
    all_subj = []
    x_axis = 1
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        top_chans = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()[:chan_num]
        all_chans = []
        for chan in top_chans:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            baseline = chan_rates['rate'].tolist()[0]
            blocks_rates = [chan_rates.iloc[i][9:14].tolist() for i in range(1, len(chan_rates))]
            y_axis = list(np.array(blocks_rates).flat)
            y_axis = np.insert(y_axis, 0, baseline)
            if len(y_axis) > x_axis:
                x_axis = len(y_axis)

            if norm == 'baseline':
                y_axis = np.array(y_axis) / baseline
                y_axis = y_axis * 100 - 100
                plt.axhline(y=0, color='r', linestyle='--')
            else:
                y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
            all_chans.append(y_axis)

        avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(y_axis))])
        means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(y_axis))]
        all_subj.append(means_subj)
        # plt.plot(list(range(0, x_axis * 5)), y_axis, '-o')
        # for i in range(0, len(chan_rates) * 5 - 1, 10):
        #     plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)

    avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, x_axis)])
    subj_num = avg_df.count().tolist()
    if type == 'even':
        means = [avg_df[str(i)].mean() for i in range(0, len(x_axis), 2)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis), 2)]
        plt.errorbar(list(range(0, len(means))), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        plt.title(f'top {chan_num} channels avg only breaks')
    elif type == 'odd':
        means = [avg_df[str(i)].mean() for i in [0] + list(range(1, len(x_axis), 2))]
        stds = [avg_df[str(i)].std(ddof=0) for i in [0] + list(range(1, len(x_axis), 2))]
        plt.errorbar(list(range(0, len(means))), means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
        plt.title(f'top {chan_num} channels avg only stim')
    else:
        means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, x_axis)]
        # plt.errorbar(list(range(0, x_axis)), means, yerr=stds, capsize=3, fmt='-o', label='avg', color='black')
        # plt.errorbar(list(range(0, x_axis))[:90], means[:90], yerr=stds[:90], capsize=3, fmt='-o', label='avg', color='black')
        plt.plot(list(range(0, x_axis))[:90], means[:90], '-o', color='black', markersize=4)
        for i in range(0, 90 - 1, 10):
        # for i in range(0, x_axis - 1, 10):
            plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)
        plt.title(f'top {chan_num} channels avg full profile')

    # for k, j in enumerate(avg_df.count().tolist()[1:]):
    #     plt.annotate(str(j), xy=(k, 150))
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.xlim(-0.2, 90)
    # plt.ylim(0, 1)
    # plt.xticks(list(range(0, len(means))))
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/top{chan_num}_chan_full_profile_avg_1min_{norm}_norm')
    plt.clf()


def top_chan_block_stim_type(subjects, chan_num=1, error=True, norm='baseline'):
    plt.rcParams['figure.figsize'] = [8, 5]
    mixed_stim = ['485', '497', '499', '505', '510-1', '510-7']
    x_axis = 1
    color = {'sync': 'black', 'mixed': 'tab:green'}
    marker = {'sync': 'o', 'mixed': '*'}
    for stim_type in ['sync', 'mixed']:
        subjects_stim = [subj for subj in subjects if (stim_type == 'mixed' and subj in mixed_stim) or
                         (stim_type == 'sync' and subj not in mixed_stim)]
        all_subj = []
        for subj in subjects_stim:
            df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
            top_chans = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()[:chan_num]
            all_chans = []
            for chan in top_chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                y_axis = chan_rates['rate'].tolist()
                # y_axis = list(np.array(blocks_rates).flat)
                # y_axis = np.insert(y_axis, 0, baseline)
                if len(y_axis) > x_axis:
                    x_axis = len(y_axis)
                if norm == 'baseline':
                    baseline = y_axis[0]
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                    plt.axhline(y=0, color='r', linestyle='--')
                elif norm == 'minmax':
                    y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
                all_chans.append(y_axis)

            avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(y_axis))])
            means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(y_axis))]
            all_subj.append(means_subj)

        avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, x_axis)])
        subj_num = avg_df.count().tolist()
        means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, x_axis)]
        if error:
            plt.errorbar(list(range(0, x_axis)), means, yerr=stds, capsize=3, fmt='-o', label=stim_type,
                         color=color[stim_type], marker=marker[stim_type])
        else:
            plt.plot(list(range(0, x_axis)), means, '-o', label=stim_type, color=color[stim_type],
                     marker=marker[stim_type])
    for i in range(0, x_axis, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.title(f'top {chan_num} channels avg full profile')


    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.legend()
    plt.xlim(-0.2, x_axis - 1)
    # plt.ylim(0, 1)
    plt.locator_params(axis='y', nbins=10)
    plt.savefig(f'results/top{chan_num}_chan_full_profile_avg_5min_stim_type_{norm}_norm')
    plt.clf()


def top_chan_minute_stim_type(subjects, chan_num=1, error=True, norm='baseline'):
    plt.rcParams['figure.figsize'] = [8, 5]
    mixed_stim = ['485', '497', '499', '505', '510-1', '510-7']
    x_axis = 1
    color = {'sync': 'black', 'mixed': 'tab:green'}
    marker = {'sync': 'o', 'mixed': '*'}
    for stim_type in ['sync', 'mixed']:
        subjects_stim = [subj for subj in subjects if (stim_type == 'mixed' and subj in mixed_stim) or
                                                      (stim_type == 'sync' and subj not in mixed_stim)]
        all_subj = []
        for subj in subjects_stim:
            df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
            top_chans = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()[:chan_num]
            all_chans = []
            for chan in top_chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                baseline = chan_rates['rate'].tolist()[0]
                blocks_rates = [chan_rates.iloc[i][9:14].tolist() for i in range(1, len(chan_rates))]
                y_axis = list(np.array(blocks_rates).flat)
                y_axis = np.insert(y_axis, 0, baseline)
                if len(y_axis) > x_axis:
                    x_axis = len(y_axis)

                if norm == 'baseline':
                    # normalization to baseline
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                elif norm == 'minmax':
                    y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
                all_chans.append(y_axis)

            avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(y_axis))])
            means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(y_axis))]
            all_subj.append(means_subj)
            # plt.plot(list(range(0, x_axis * 5)), y_axis, '-o')
            # for i in range(0, len(chan_rates) * 5 - 1, 10):
            #     plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)

        avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, x_axis)])
        subj_num = avg_df.count().tolist()
        means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, x_axis)]
        # plt.errorbar(list(range(0, x_axis)), means, yerr=stds, capsize=3, fmt='-o', label='avg', color='black')
        if error:
            plt.errorbar(list(range(0, x_axis))[:90], means[:90], yerr=stds[:90], capsize=3, fmt='-o', label=stim_type,
                         color=color[stim_type], marker=marker[stim_type])
        else:
            plt.plot(list(range(0, x_axis))[:90], means[:90], '-o', label=stim_type, color=color[stim_type], marker=marker[stim_type])
    for i in range(0, 90 - 1, 10):
        # for i in range(0, x_axis - 1, 10):
        plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)
    plt.title(f'top {chan_num} channels avg full profile')

    # for k, j in enumerate(avg_df.count().tolist()[1:]):
    #     plt.annotate(str(j), xy=(k, 150))
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.legend()
    plt.xlim(-0.2, 90)
    # plt.ylim(0, 1)
    # plt.xticks(list(range(0, len(means))))
    plt.locator_params(axis='y', nbins=10)
    plt.savefig(f'results/top{chan_num}_chan_full_profile_avg_1min_stim_type_{norm}_norm')


def top_channels_plot(subjects, chan_num=1):
    all_chans = {}
    hemisphere = {'R': 0, 'L': 0}
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        top_chans = df.sort_values(by='baseline', ascending=False).iloc[:10, :]['channel'].tolist()[:chan_num]
        for chan in top_chans:
            hemisphere[chan[0]] += 1
            # take only the contact name, no side and no number
            curr = chan[1:-1]
            if curr not in all_chans:
                all_chans[curr] = 1
            else:
                all_chans[curr] += 1

    sorted_chans = dict(sorted(all_chans.items(), key=lambda item: item[1]))
    plt.barh(*zip(*sorted_chans.items()))
    # plt.barh(*zip(*hemisphere.items()))
    plt.yticks(fontsize=10)
    plt.show()
    print(1)


def top_chan_minute_profile_avg_hemisphere(subjects, chan_num=10):
    stim_side_right = ['485', '487', '489', '490', '496', '497', '510-1', '510-7', '515', '520', '538', '544', '545']
    plt.rcParams['figure.figsize'] = [8, 5]
    x_axis = 1
    color = {'ipsi': 'black', 'contra': 'tab:green'}
    marker = {'ipsi': 'o', 'contra': '*'}
    for stim_type in ['ipsi', 'contra']:
        all_subj, baselines = [], []
        for subj in subjects:
            if stim_type == 'ipsi':
                curr_side = 'R' if subj in stim_side_right else 'L'
            else:
                curr_side = 'R' if subj not in stim_side_right else 'L'
            df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
            top_chans = df[df['channel'].str.startswith(curr_side)].sort_values(by='baseline', ascending=False).iloc[:10, :]
            # remove baseline 0
            top_chans = top_chans.drop(top_chans[top_chans.baseline == 0].index)['channel'].tolist()[:chan_num]
            all_chans = []
            for chan in top_chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                baseline = chan_rates['rate'].tolist()[0]
                baselines.append(baseline)
                blocks_rates = [chan_rates.iloc[i][9:14].tolist() for i in range(1, len(chan_rates))]
                y_axis = list(np.array(blocks_rates).flat)
                y_axis = np.insert(y_axis, 0, baseline)
                if len(y_axis) > x_axis:
                    x_axis = len(y_axis)

                # normalization to baseline
                y_axis = np.array(y_axis) / baseline
                y_axis = y_axis * 100 - 100
                all_chans.append(y_axis)

            avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(y_axis))])
            means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(y_axis))]
            all_subj.append(means_subj)
            # plt.plot(list(range(0, x_axis * 5)), y_axis, '-o')
            # for i in range(0, len(chan_rates) * 5 - 1, 10):
            #     plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)

        avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, x_axis)])
        subj_num = avg_df.count().tolist()
        means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, x_axis)]
        # plt.errorbar(list(range(0, x_axis)), means, yerr=stds, capsize=3, fmt='-o', label='avg', color='black')
        # plt.errorbar(list(range(0, x_axis))[:90], means[:90], yerr=stds[:90], capsize=3, fmt='-o',
        #              label=f'{stim_type}- avg baseline: {np.mean(np.array(baselines))}',
        #              color=color[stim_type], marker=marker[stim_type])
        plt.plot(list(range(0, x_axis))[:90], means[:90], '-o', label=stim_type, color=color[stim_type],
                 marker=marker[stim_type])
    for i in range(0, 90 - 1, 10):
        # for i in range(0, x_axis - 1, 10):
        plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)
    plt.title(f'top {chan_num} channels avg full profile')

    # for k, j in enumerate(avg_df.count().tolist()[1:]):
    #     plt.annotate(str(j), xy=(k, 150))
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.legend()
    plt.xlim(-0.2, 90)
    # plt.ylim(0, 1)
    # plt.xticks(list(range(0, len(means))))
    plt.locator_params(axis='y', nbins=10)
    plt.savefig(f'top{chan_num}_chan_full_profile_avg_1min_base_norm_hemisphere')

def top_chan_block_profile_avg_hemisphere(subjects, chan_num=10, norm='baseline'):
    stim_side_right = ['485', '487', '489', '490', '496', '497', '510-1', '510-7', '515', '520', '538', '544', '545']
    plt.rcParams['figure.figsize'] = [8, 5]
    x_axis = 1
    color = {'ipsi': 'black', 'contra': 'tab:green'}
    marker = {'ipsi': 'o', 'contra': '*'}
    for stim_type in ['ipsi', 'contra']:
        all_subj, baselines = [], []
        for subj in subjects:
            if stim_type == 'ipsi':
                curr_side = 'R' if subj in stim_side_right else 'L'
            else:
                curr_side = 'R' if subj not in stim_side_right else 'L'
            df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
            top_chans = df[df['channel'].str.startswith(curr_side)].sort_values(by='baseline', ascending=False).iloc[:10, :]
            # remove baseline 0
            top_chans = top_chans.drop(top_chans[top_chans.baseline == 0].index)['channel'].tolist()[:chan_num]
            all_chans = []
            for chan in top_chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                y_axis = chan_rates['rate'].tolist()
                # y_axis = list(np.array(blocks_rates).flat)
                # y_axis = np.insert(y_axis, 0, baseline)
                if len(y_axis) > x_axis:
                    x_axis = len(y_axis)
                if norm == 'baseline':
                    baseline = y_axis[0]
                    y_axis = np.array(y_axis) / baseline
                    y_axis = y_axis * 100 - 100
                    plt.axhline(y=0, color='r', linestyle='--')
                elif norm == 'minmax':
                    y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
                all_chans.append(y_axis)

            avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(y_axis))])
            means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(y_axis))]
            all_subj.append(means_subj)
            # plt.plot(list(range(0, x_axis * 5)), y_axis, '-o')
            # for i in range(0, len(chan_rates) * 5 - 1, 10):
            #     plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)

        avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, x_axis)])
        subj_num = avg_df.count().tolist()
        means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, x_axis)]
        # plt.errorbar(list(range(0, x_axis)), means, yerr=stds, capsize=3, fmt='-o', label='avg', color='black')
        # plt.errorbar(list(range(0, x_axis))[:90], means[:90], yerr=stds[:90], capsize=3, fmt='-o',
        #              label=f'{stim_type}- avg baseline: {np.mean(np.array(baselines))}',
        #              color=color[stim_type], marker=marker[stim_type])
        plt.plot(list(range(0, x_axis)), means, '-o', label=stim_type, color=color[stim_type],
                 marker=marker[stim_type])
    for i in range(0, x_axis, 2):
        # for i in range(0, x_axis - 1, 10):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.title(f'top {chan_num} channels avg full profile')

    # for k, j in enumerate(avg_df.count().tolist()[1:]):
    #     plt.annotate(str(j), xy=(k, 150))
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.legend()
    plt.xlim(-0.2, x_axis)
    # plt.ylim(0, 1)
    # plt.xticks(list(range(0, len(means))))
    plt.locator_params(axis='y', nbins=10)
    plt.savefig(f'top{chan_num}_chan_full_profile_avg_1min_base_norm_hemisphere')


def stim_profile(subjects, plot_subj=False, norm=False, save=False):
    all_subj = []
    x_axis = 1
    for subj in subjects:
        stims_count = {'n_stim': [], 'duration_sec': [], 'rate': [], 'duration_20%': [],
                       'n_1_20%': [], 'n_2_20%': [], 'n_3_20%': [], 'n_4_20%': [], 'n_5_20%': [],
                       'rate_1_20%': [], 'rate_2_20%': [], 'rate_3_20%': [], 'rate_4_20%': [], 'rate_5_20%': []}
        stim = pd.read_csv(stim_path % (subj, subj), header=None).iloc[0, :].to_list()
        stim = [round(x) for x in stim]
        stim_sections_sec = get_stim_starts(subj)
        for i, (start, end) in enumerate(stim_sections_sec):
            # get the relevant section of the list with all stimuli
            start_round = start * 1000 if start * 1000 in stim else round(start * 1000)
            end_round = end * 1000 if end * 1000 in stim else round(end * 1000)
            current = stim[len(stim) - stim[::-1].index(start_round) - 1: stim.index(end_round) + 1]
            stims_count['n_stim'].append(len(current))
            stims_count['duration_sec'].append((end - start))
            stims_count['rate'].append(stims_count['n_stim'][i] / (stims_count['duration_sec'][i] / 60))
            duration_20_sec = stims_count['duration_sec'][i] / 5
            stims_count['duration_20%'].append(duration_20_sec)
            for j in range(1, 6):
                # get the first and last timestamp possible for that section
                start_20 = start + (j - 1) * duration_20_sec
                end_20 = start + j * duration_20_sec
                # find the specific values in the list that are in between the timestamps
                start_20_val = [current[i] for i, x in enumerate(current) if i < len(current) and current[i] >= start_20 * 1000][0]
                end_20_val_list = [x for i, x in enumerate(reversed(current)) if i + 1 < len(current) and x <= end_20 * 1000]
                # sometimes there is only one stim and that list turns out as empty
                if len(end_20_val_list) > 0:
                    end_20_val = end_20_val_list[0]
                    n_20 = len(current[current.index(start_20_val): current.index(end_20_val) + 1])
                else:
                    n_20 = 1
                stims_count[f'n_{str(j)}_20%'].append(n_20)
                stims_count[f'rate_{str(j)}_20%'].append(n_20 / (duration_20_sec / 60))

        stim_df = pd.DataFrame(stims_count)
        blocks_rates = [stim_df.iloc[i][9:14].tolist() for i in range(0, len(stim_df))]
        y_axis = list(np.array(blocks_rates).flat)
        if len(y_axis) > x_axis:
            x_axis = len(y_axis)
        if norm:
            y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
        all_subj.append(y_axis)

        if plot_subj:
            plt.plot(list(range(len(y_axis))), y_axis, '-o')
            for i in range(0, len(y_axis) - 1, 10):
                plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)
            plt.title(f'{subj} stim rate full profile')
            plt.xlabel('Time point')
            plt.ylabel('stim rate')
            plt.xlim(-0.2, len(y_axis) - 1)
            # plt.xlim(-0.2, len(y_axis) - 1)
            # plt.locator_params(axis='y', nbins=8)
            # plt.savefig(f'results/{subj}_stim_rates.png')
            # plt.clf()
            plt.show()
        if save:
            stim_df.to_csv(f'results/{subj}_stim_rates.csv')

    if not plot_subj:
        avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, x_axis)])
        means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, x_axis)]
        # plt.errorbar(list(range(0, x_axis)), means, yerr=stds, capsize=3, fmt='-o', label='avg', color='black')
        plt.errorbar(list(range(0, x_axis)), means, yerr=stds, capsize=3, fmt='-o', label='avg',
                     color='black')
        for i in range(0, x_axis - 1, 10):
            plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)
        plt.title(f'stim rate full profile')

        # for k, j in enumerate(avg_df.count().tolist()[1:]):
        #     plt.annotate(str(j), xy=(k, 150))
        plt.xlabel('Time point')
        plt.ylabel('stim rate')
        plt.xlim(-0.2, len(means) - 1)
        # plt.ylim(0, 1)
        # plt.xticks(list(range(0, len(means))))
        plt.locator_params(axis='y', nbins=15)
        plt.savefig('stim_rates_avg_1_min_norm')
        print(1)


def stim_profile_5_min(subjects, norm=False, plot_subj=True):
    all_subj = []
    x_axis = []
    for subj in subjects:
        stim_rates = pd.read_csv(f'results\\{subj}_stim_rates.csv', index_col=0)
        if len(stim_rates) > len(x_axis):
            x_axis = list(range(0, len(stim_rates)))
        y_axis = stim_rates['rate'].tolist()
        if norm:
            y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
        all_subj.append(y_axis)
        if plot_subj:
            plt.plot(list(range(1, len(y_axis) + 1)), y_axis, '-o')
            # for i in range(0, len(y_axis) - 1, 2):
            #     plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
            plt.title(f'stim rate full profile')
            plt.xlabel('Time point')
            plt.ylabel('stim rate')
            plt.xlim(0.5, len(x_axis))
            # plt.show()
            plt.savefig(f'results/{subj}_stim_rates_5_min.png')
            plt.clf()

    avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, len(x_axis))])
    means = [avg_df[str(i)].mean() for i in range(0, len(x_axis))]
    stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis))]
    plt.errorbar(x_axis, means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
    # for i in range(0, len(x_axis) - 1, 2):
    #     plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.title(f'stim rate 5 min profile')

    # for k, j in enumerate(avg_df.count().tolist()[1:]):
    #     plt.annotate(str(j), xy=(x_axis[k], 150))
    plt.xlabel('Time point')
    plt.ylabel('stim rate')
    plt.xlim(-0.2, len(means) - 1)
    # plt.ylim(0, 1)
    plt.xticks(list(range(0, len(means))))
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/all_stim_rates_5_min.png')

def temporal_profile_avg(subjects, norm='baseline', chans=['AH1', 'MH1', 'RA1', 'LA1', 'PHG1', 'EC1'], plot_subj=False):
    all_subj = []
    x_axis = []
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        frontal = df[df['channel'].str.contains('OF1')]['channel'].tolist()
        curr_chans = df[df['channel'].str.contains('|'.join(chans))]['channel'].tolist()
        all_chans = []
        final_chans = []
        for chan in curr_chans:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            y_axis = chan_rates['rate'].tolist()
            # only channels with baseline rate higher than 0
            if y_axis[0] > 0:
                final_chans.append(chan)
                if len(chan_rates) > len(x_axis):
                    x_axis = list(range(0, len(chan_rates)))
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / y_axis[0]
                    y_axis = y_axis * 100 - 100
                elif norm == 'minmax':
                    y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
                all_chans.append(y_axis)
        avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(chan_rates))])
        means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(chan_rates))]
        std_subj = [avg_df_subj[str(i)].std(ddof=0) for i in range(0, len(chan_rates))]
        all_subj.append(means_subj)
        if plot_subj:
            chans_text = ', '.join(final_chans)
            plt.gcf().text(0.15, 0.01, chans_text, fontsize=10)
            plt.errorbar(range(len(y_axis)), means_subj, yerr=std_subj, capsize=5, fmt='-o', label='avg', color='black')
            for i in range(0, len(x_axis) - 1, 2):
                plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
            plt.title(f'{subj} temporal channels avg full profile')
            plt.axhline(y=0, color='r', linestyle='--')
            # plt.xlabel('Time point')
            plt.ylabel('% change in spike rate')
            plt.xlim(-0.2, len(y_axis) - 1)
            plt.xticks(list(range(0, len(means_subj))))
            plt.locator_params(axis='y', nbins=15)
            plt.savefig(f'results/{subj}_temporal_full_profile_avg_5min_{norm}_norm')
            plt.clf()

    avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, len(x_axis))])
    subj_num = avg_df.count().tolist()

    means = [avg_df[str(i)].mean() for i in range(0, len(x_axis))]
    stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis))]
    plt.errorbar(x_axis, means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
    for i in range(0, len(x_axis) - 1, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.title(f'temporal channels avg full profile')

    for k, j in enumerate(avg_df.count().tolist()[1:]):
        plt.annotate(str(j), xy=(x_axis[k], 150))
    if norm == 'baseline':
        plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.xlim(-0.2, len(means) - 1)
    # plt.ylim(0, 1)
    plt.xticks(list(range(0, len(means))))
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/temporal_chan_full_profile_avg_5min_{norm}_norm')
    plt.clf()
    # plt.show()

def frontal_profile_avg(subjects, norm='baseline', plot_subj=False):
    all_subj = []
    x_axis = []
    for subj in subjects:
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        curr_chans = df[df['channel'].str.contains('F')]['channel'].tolist()
        all_chans = []
        final_chans = []
        for chan in curr_chans:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            y_axis = chan_rates['rate'].tolist()
            # only channels with baseline rate higher than 0
            if y_axis[0] > 0:
                final_chans.append(chan)
                if len(chan_rates) > len(x_axis):
                    x_axis = list(range(0, len(chan_rates)))
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / y_axis[0]
                    y_axis = y_axis * 100 - 100
                elif norm == 'minmax':
                    y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
                all_chans.append(y_axis)
        avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(chan_rates))])
        means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(chan_rates))]
        std_subj = [avg_df_subj[str(i)].std(ddof=0) for i in range(0, len(chan_rates))]
        all_subj.append(means_subj)
        if plot_subj:
            chans_text = ', '.join(final_chans)
            plt.gcf().text(0.15, 0.01, chans_text, fontsize=10)
            plt.errorbar(range(len(y_axis)), means_subj, yerr=std_subj, capsize=5, fmt='-o', label='avg', color='black')
            for i in range(0, len(x_axis) - 1, 2):
                plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
            plt.title(f'{subj} frontal channels avg full profile')
            plt.axhline(y=0, color='r', linestyle='--')
            # plt.xlabel('Time point')
            plt.ylabel('% change in spike rate')
            plt.xlim(-0.2, len(y_axis) - 1)
            plt.xticks(list(range(0, len(means_subj))))
            plt.locator_params(axis='y', nbins=15)
            plt.savefig(f'results/{subj}_frontal_full_profile_avg_5min_{norm}_norm')
            plt.clf()

    avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, len(x_axis))])
    subj_num = avg_df.count().tolist()

    means = [avg_df[str(i)].mean() for i in range(0, len(x_axis))]
    stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis))]
    plt.errorbar(x_axis, means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
    for i in range(0, len(x_axis) - 1, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.title(f'frontal channels avg full profile')

    for k, j in enumerate(avg_df.count().tolist()[1:]):
        plt.annotate(str(j), xy=(x_axis[k], 150))
    if norm == 'baseline':
        plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.xlim(-0.2, len(means) - 1)
    # plt.ylim(0, 1)
    plt.xticks(list(range(0, len(means))))
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/frontal_chan_full_profile_avg_5min_{norm}_norm')
    plt.clf()
    # plt.show()


def temporal_vs_frontal(subjects, norm='baseline', plot_subj=False):
    plt.rcParams['figure.figsize'] = [8, 5]
    color = {'temporal': 'black', 'frontal': 'tab:green'}
    marker = {'temporal': 'o', 'frontal': '*'}
    all_subj, x_axis = [], []
    for area in ['temporal', 'frontal']:
        for subj in subjects:
            df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
            if area == 'temporal':
                curr_chans = df[df['channel'].str.contains('|'.join(['AH1', 'MH1', 'RA1', 'LA1', 'PHG1', 'EC1']))]['channel'].tolist()
            else:
                curr_chans = df[df['channel'].str.contains('OF1')]['channel'].tolist()
            all_chans, final_chans = [], []
            for chan in curr_chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                y_axis = chan_rates['rate'].tolist()
                # only channels with baseline rate higher than 0
                if y_axis[0] > 0:
                    final_chans.append(chan)
                    if len(chan_rates) > len(x_axis):
                        x_axis = list(range(0, len(chan_rates)))
                    if norm == 'baseline':
                        y_axis = np.array(y_axis) / y_axis[0]
                        y_axis = y_axis * 100 - 100
                    elif norm == 'minmax':
                        y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
                    all_chans.append(y_axis)
            avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(chan_rates))])
            means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(chan_rates))]
            std_subj = [avg_df_subj[str(i)].std(ddof=0) for i in range(0, len(chan_rates))]
            all_subj.append(means_subj)
            if plot_subj:
                chans_text = ', '.join(final_chans)
                plt.gcf().text(0.15, 0.01, chans_text, fontsize=10)
                plt.errorbar(range(len(y_axis)), means_subj, yerr=std_subj, capsize=5, fmt='-o', label='avg',
                             color='black')
                for i in range(0, len(x_axis) - 1, 2):
                    plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
                plt.title(f'{subj} temporal channels avg full profile')
                plt.axhline(y=0, color='r', linestyle='--')
                # plt.xlabel('Time point')
                plt.ylabel('% change in spike rate')
                plt.xlim(-0.2, len(y_axis) - 1)
                plt.xticks(list(range(0, len(means_subj))))
                plt.locator_params(axis='y', nbins=15)
                plt.savefig(f'results/{subj}_temporal_full_profile_avg_5min_{norm}_norm')
                plt.clf()

        avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, len(x_axis))])
        subj_num = avg_df.count().tolist()

        means = [avg_df[str(i)].mean() for i in range(0, len(x_axis))]
        stds = [avg_df[str(i)].std(ddof=0) for i in range(0, len(x_axis))]
        plt.errorbar(x_axis, means, yerr=stds, capsize=5, fmt='-o', label=area, color=color[area])
    for i in range(0, len(x_axis) - 1, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.title(f'temporal vs frontal channels avg full profile')

    # for k, j in enumerate(avg_df.count().tolist()[1:]):
    #     plt.annotate(str(j), xy=(x_axis[k], 150))
    if norm == 'baseline':
        plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.xlim(-0.2, len(means) - 1)
    # plt.ylim(0, 1)
    plt.xticks(list(range(0, len(means))))
    plt.legend()
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/temporal_vs_frontal_full_profile_avg_5min_{norm}_norm')
    plt.clf()

# calc rate per chan and plot top ten
# all = ['485', '486', '487', '488', '489', '490', '496', '497', '498', '499', '505', '510', '515', '520', '538',
#        '541', '544', '545']
# subjects = ['485', '486', '487', '488', '489', '496', '497', '498', '505', '515', '538', '541', '544', '545']
frontal_stim = ['485', '486', '487', '488', '497', '498', '499', '505', '510-1', '510-7', '515', '520', '545']
# temporal_stim = ['489', '490', '496', '538']
mixed_stim = ['485', '497', '499', '505', '510-1', '510-7']
all_except_490_520 = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '538', '541', '544', '545']
all_after_clean = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '520', '538', '545']
# calc_rate_per_chan(all_except_490_520)
# top_channel_full_profile(subjects=['488'])
# top_channel_full_profile(all_except_490_520)

# top_chan_profile_avg(['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '538', '541', '544', '545'], type='odd')

# top_chan_minute_profile_avg(['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '538', '541', '544', '545'], norm='mix', chan_num=10)
# top_chan_minute_stim_type(['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '538', '541', '544', '545'], chan_num=1)
# top_channels_plot(['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '538', '541', '544', '545'])
# top_chan_minute_profile_avg_hemisphere(['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '538', '541', '544', '545'])
# stim_profile(all_except_490_520, plot_subj=False)
# stim_profile_5_min(all_except_490_520)

# round 4
# calc_rate_per_chan(['541', '544'])
# top_channel_full_profile(all_after_clean, norm=None)
# top10_chan_profile_avg(['485', '486', '489', '496', '497', '498', '505', '515', '520'], chan_num=1) # manually change filename
# top10_chan_profile_avg(['485', '486', '489', '496', '497', '498', '505', '515', '520'], chan_num=1, norm='minmax') # manually change filename
# top10_chan_profile_avg(all_after_clean, chan_num=1)
# top10_chan_profile_avg(all_after_clean, chan_num=10)
# top10_chan_profile_avg(all_after_clean, chan_num=1, norm='minmax')
# top10_chan_profile_avg(all_after_clean, chan_num=10, norm='minmax')
# top10_chan_profile_avg(all_after_clean, chan_num=1, norm='baseline', plot_subj=True)
# top10_chan_profile_avg(all_after_clean, chan_num=10, norm='baseline', plot_subj=True)
# top_chan_minute_profile_avg(all_after_clean)
# top_chan_minute_profile_avg(all_after_clean, norm='minmax')
# top_chan_minute_profile_avg(all_after_clean, chan_num=10)
# top_chan_minute_profile_avg(all_after_clean, chan_num=10, norm='minmax')
# top_chan_minute_stim_type(all_after_clean, chan_num=1, error=False)
# top_chan_minute_stim_type(all_after_clean, chan_num=10, error=False)
# top_chan_block_stim_type(all_after_clean, chan_num=1, error=False)
# top_chan_block_stim_type(all_after_clean, chan_num=10, error=False)
# top_chan_block_stim_type(all_after_clean, chan_num=10, error=False, norm='minmax')
# top_chan_block_stim_type(all_after_clean, chan_num=1, error=False, norm='minmax')
# temporal_profile_avg(all_after_clean, plot_subj=True)
# top_channels_plot(all_after_clean)
# top_channels_plot(all_after_clean, chan_num=10)
# top_chan_minute_profile_avg_hemisphere(all_after_clean)
# top_chan_block_profile_avg_hemisphere(all_after_clean)
# stim_profile(all_after_clean, plot_subj=False, save=True)
# stim_profile_5_min(all_after_clean)
# temporal_profile_avg(all_after_clean, plot_subj=False)
# frontal_profile_avg(all_after_clean, plot_subj=True)
# temporal_vs_frontal(all_after_clean, plot_subj=False)
# temporal_vs_frontal(frontal_stim, plot_subj=False)

manual_select = {'485': ['RMH1, RPHG1', 'RBAA1'], '489': ['LPHG2', 'RAH1', 'RPHG1'], '496': ['RMH1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'], '515': ['RA1', 'RAH2'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}

# control round
calc_rate_per_chan(['396', '398', '402', '406', '415', '416'], control=True)