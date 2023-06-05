import mne
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob
from utils import get_stim_starts, edf_path, stim_path
from scipy.stats import sem


def top_chan_profile_avg(subjects, norm='baseline', chan_num=10, plot_subj=False):
    plt.rcParams['figure.figsize'] = [8, 5]
    all_subj = []
    x_axis = []
    for subj in subjects.keys():
        top_chans = subjects[subj][:chan_num]
        all_chans = []
        for chan in top_chans:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
            # check the x axis for the avg of all subjects
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
        std_subj = [sem(avg_df_subj[str(i)], nan_policy='omit') for i in range(0, len(chan_rates))]
        all_subj.append(means_subj)
        if plot_subj:
            if chan_num == 1:
                plt.title(f'{subj} top channel {top_chans[0]} avg full profile')
            else:
                plt.title(f'{subj} top {chan_num} channels avg full profile')
                chans_text = ', '.join(top_chans)
                plt.gcf().text(0.15, 0.01, chans_text, fontsize=10)
            plt.errorbar(range(len(y_axis)), means_subj, yerr=std_subj, capsize=5, fmt='-o', label='avg', color='black')
            plt.xlim(-0.8, len(y_axis) - 1)
            for i in range(0, len(x_axis) - 1, 2):
                plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
            sleep_percent = pd.read_csv(f'results/{subj}_sleep_percent.csv')
            for k, j in enumerate(sleep_percent['sleep_percent']):
                plt.text(k -0.7, max(means_subj) - 2, str(int(j)), size=8)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.ylabel('% change in spike rate')

            plt.xticks(list(range(0, len(means_subj))))
            plt.locator_params(axis='y', nbins=15)
            plt.savefig(f'results/{subj}_top_chans_full_profile_avg_5min_{norm}_norm')
            plt.clf()

    avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, len(x_axis))])
    means = [avg_df[str(i)].mean() for i in range(0, len(x_axis))]
    stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, len(x_axis))]
    plt.errorbar(x_axis, means, yerr=stds, capsize=5, fmt='-o', label='avg', color='black')
    for i in range(0, len(x_axis) - 1, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.title(f'top channels avg full profile')

    for k, j in enumerate(avg_df.count().tolist()[1:]):
        plt.text(k, plt.ylim()[1] - 5, str(j), size=8)
    if norm == 'baseline':
        plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.xlim(-0.2, len(means) - 1)
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/top_chans_full_profile_avg_5min_{norm}_norm')
    plt.clf()


def top_chan_minute_profile_avg(subjects, norm='baseline', chan_num=1):
    plt.rcParams['figure.figsize'] = [8, 5]
    all_subj = []
    x_axis = 1
    for subj in subjects.keys():
        top_chans = subjects[subj]
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

    avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, x_axis)])
    means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
    plt.plot(list(range(0, x_axis))[:90], means[:90], '-o', color='black', markersize=4)
    for i in range(0, 90 - 1, 10):
        plt.axvspan(i, i + 5, facecolor='silver', alpha=0.5)
    plt.title(f'top channels avg full profile')

    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.xlim(-0.2, 90)
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/top_chan_full_profile_avg_1min_{norm}_norm')
    plt.clf()


def chans_profile_separate(subjects, norm='baseline', filename='top', csv_file='results\\%s_%s_rates.csv'):
    plt.rcParams['figure.figsize'] = [8, 5]
    x_axis = []
    for subj in subjects.keys():
        top_chans = subjects[subj]
        all_chans = []
        for chan in top_chans:
            chan_rates = pd.read_csv(csv_file % (subj, chan), index_col=0)
            # check the x axis for the avg of all subjects
            if len(chan_rates) > len(x_axis):
                x_axis = list(range(0, len(chan_rates)))
            y_axis = chan_rates['rate'].tolist()
            if norm == 'baseline':
                y_axis = np.array(y_axis) / y_axis[0]
                y_axis = y_axis * 100 - 100
            elif norm == 'minmax':
                y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
            all_chans.append(y_axis)
            plt.plot(range(len(y_axis)), y_axis, '-o', label=chan)
        avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(chan_rates))])
        means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(chan_rates))]
        std_subj = [sem(avg_df_subj[str(i)]) for i in range(0, len(chan_rates))]
        plt.title(f'{subj} {filename} channels full profile')
        plt.errorbar(range(len(y_axis)), means_subj, yerr=std_subj, capsize=5, fmt='-o', label='avg', color='black')
        plt.xlim(-0.8, len(y_axis) - 1)
        for i in range(0, len(x_axis) - 1, 2):
            plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
        sleep_percent = pd.read_csv(f'results/{subj}_sleep_percent.csv')
        for k, j in enumerate(sleep_percent['sleep_percent']):
            plt.text(k -0.7, plt.ylim()[1] - 5, str(int(j)), size=8)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.legend(loc='lower right')
        plt.ylabel('% change in spike rate')
        plt.xticks(list(range(0, len(means_subj))))
        plt.locator_params(axis='y', nbins=15)
        plt.savefig(f'results/{subj}_{filename}_chans_full_profile_separate_5min_{norm}_norm')
        plt.clf()


def top_chan_block_stim_type(subjects, error=True, norm='baseline'):
    plt.rcParams['figure.figsize'] = [8, 5]
    mixed_stim = ['485', '497', '499', '505', '510-1', '510-7']
    x_axis = 1
    color = {'sync': 'black', 'mixed': 'tab:green'}
    marker = {'sync': 'o', 'mixed': '*'}
    for stim_type in ['sync', 'mixed']:
        subjects_stim = [subj for subj in subjects.keys() if (stim_type == 'mixed' and subj in mixed_stim) or
                         (stim_type == 'sync' and subj not in mixed_stim)]
        all_subj = []
        for subj in subjects_stim:
            top_chans = subjects[subj]
            all_chans = []
            for chan in top_chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                y_axis = chan_rates['rate'].tolist()
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
        means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
        stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, x_axis)]

        if error:
            plt.errorbar(list(range(0, x_axis)), means, yerr=stds, capsize=3, fmt='-o', label=stim_type,
                         color=color[stim_type], marker=marker[stim_type])
        else:
            plt.plot(list(range(0, x_axis)), means, '-o', label=stim_type, color=color[stim_type],
                     marker=marker[stim_type])
        for k, j in enumerate(avg_df.count().tolist()[1:]):
            if stim_type == 'sync':
                plt.text(k, plt.ylim()[1] - 5, str(j), size=8, color=color[stim_type])
            else:
                plt.text(k, plt.ylim()[1] - 15, str(j), size=8, color=color[stim_type])

    for i in range(0, x_axis, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.title(f'top channels avg full profile')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.legend(loc='lower right')
    plt.xlim(-0.2, x_axis - 1)
    # plt.ylim(0, 1)
    plt.locator_params(axis='y', nbins=10)
    plt.savefig(f'results/top_chans_full_profile_avg_5min_stim_type_{norm}_norm')
    plt.clf()


def top_chan_block_profile_avg_hemisphere(subjects, chan_num=10, norm='baseline'):
    stim_side_right = ['485', '487', '489', '490', '496', '497', '510-1', '510-7', '515', '520', '538', '544', '545']
    plt.rcParams['figure.figsize'] = [8, 5]
    x_axis = 1
    color = {'ipsi': 'black', 'contra': 'tab:green'}
    marker = {'ipsi': 'o', 'contra': '*'}
    for stim_type in ['ipsi', 'contra']:
        all_subj, baselines = [], []
        for subj in subjects.keys():
            if stim_type == 'ipsi':
                curr_side = 'R' if subj in stim_side_right else 'L'
            else:
                curr_side = 'R' if subj not in stim_side_right else 'L'
            top_chans = [x for x in subjects[subj] if x[0] == curr_side]
            all_chans = []
            for chan in top_chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                y_axis = chan_rates['rate'].tolist()
                if len(y_axis) > x_axis:
                    x_axis = len(y_axis)
                if norm == 'baseline':
                    baseline = y_axis[0]
                    baselines.append(baseline)
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
        means = [avg_df[str(i)].mean() for i in range(0, x_axis)]
        stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, x_axis)]
        plt.errorbar(list(range(0, x_axis))[:90], means[:90], yerr=stds[:90], capsize=3, fmt='-o',
                     label=f'{stim_type}- avg baseline: {round(np.mean(np.array(baselines)), 2)}',
                     color=color[stim_type], marker=marker[stim_type])
        for k, j in enumerate(avg_df.count().tolist()[1:]):
            if stim_type == 'ipsi':
                plt.text(k, plt.ylim()[1] - 5, str(j), size=8, color=color[stim_type])
            else:
                plt.text(k, plt.ylim()[1] - 10, str(j), size=8, color=color[stim_type])
    for i in range(0, x_axis, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)

    plt.title(f'top channels avg full profile')
    plt.xlabel('Time point')
    plt.ylabel('% change in spike rate')
    plt.legend(loc='lower right')
    plt.xlim(-0.2, x_axis)
    plt.locator_params(axis='y', nbins=10)
    plt.savefig(f'results/top_chans_full_profile_avg_block_base_norm_hemisphere')


def subj_lateral_vs_mesial(lateral, mesial, norm='baseline'):
    # check 1-3 vs. 5-7
    plt.rcParams['figure.figsize'] = [8, 5]
    x_axis = []
    color = {'lateral': 'black', 'mesial': 'tab:green'}
    subjects = [subj for subj in lateral.keys() if subj in mesial.keys()]
    for subj in subjects:
        for type in color.keys():
            chans = lateral[subj] if type == 'lateral' else mesial[subj]
            all_chans = []
            for chan in chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                # check the x axis for the avg of all subjects
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
            std_subj = [sem(avg_df_subj[str(i)]) for i in range(0, len(chan_rates))]
            chans_text = ', '.join(chans)
            plt.gcf().text(0.15 if type == 'lateral' else 0.4, 0.01, chans_text, fontsize=10, color=color[type])
            plt.errorbar(range(len(y_axis)), means_subj, yerr=std_subj, capsize=5, fmt='-o', label=type, color=color[type])
        plt.title(f'{subj} lateral vs mesial avg channels full profile')
        plt.xlim(-0.8, len(y_axis) - 1)
        for i in range(0, len(x_axis) - 1, 2):
            plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
        sleep_percent = pd.read_csv(f'results/{subj}_sleep_percent.csv')
        for k, j in enumerate(sleep_percent['sleep_percent']):
            plt.text(k - 0.7, plt.ylim()[1] - 5, str(int(j)), size=8)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.legend(loc='lower right')
        plt.ylabel('% change in spike rate')
        plt.xticks(list(range(0, len(means_subj))))
        plt.locator_params(axis='y', nbins=15)
        plt.savefig(f'results/{subj}_tempral_vs_mesial_avg_chans_full_profile_5min_{norm}_norm')
        plt.clf()

def lateral_vs_mesial(lateral, mesial, norm='baseline'):
    # check 1-3 vs. 5-7
    plt.rcParams['figure.figsize'] = [8, 5]
    x_axis = []
    color = {'lateral': 'black', 'mesial': 'tab:green'}
    for type in color.keys():
        all_subj = []
        baselines = []
        subjects = lateral.keys() if type == 'lateral' else mesial.keys()
        for subj in subjects:
            chans = lateral[subj] if type == 'lateral' else mesial[subj]
            all_chans = []
            for chan in chans:
                chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv', index_col=0)
                # check the x axis for the avg of all subjects
                if len(chan_rates) > len(x_axis):
                    x_axis = list(range(0, len(chan_rates)))
                y_axis = chan_rates['rate'].tolist()
                baselines.append(y_axis[0])
                if norm == 'baseline':
                    y_axis = np.array(y_axis) / y_axis[0]
                    y_axis = y_axis * 100 - 100
                elif norm == 'minmax':
                    y_axis = (y_axis - np.min(y_axis)) / np.ptp(y_axis)
                all_chans.append(y_axis)
            avg_df_subj = pd.DataFrame(all_chans, columns=[str(x) for x in range(0, len(chan_rates))])
            means_subj = [avg_df_subj[str(i)].mean() for i in range(0, len(chan_rates))]
            all_subj.append(means_subj)

        avg_df = pd.DataFrame(all_subj, columns=[str(x) for x in range(0, len(x_axis))])
        means = [avg_df[str(i)].mean() for i in range(0, len(x_axis))]
        stds = [sem(avg_df[str(i)], nan_policy='omit') for i in range(0, len(x_axis))]
        plt.errorbar(list(range(0, len(x_axis))), means, yerr=stds, capsize=3, fmt='-o',
                     label=f'{type}- avg baseline: {round(np.mean(np.array(baselines)), 2)}', color=color[type])
        for k, j in enumerate(avg_df.count().tolist()[1:]):
            if type == 'lateral':
                plt.text(k, plt.ylim()[1] - 5, str(j), size=8, color=color[type])
            else:
                plt.text(k, plt.ylim()[1] - 10, str(j), size=8, color=color[type])
    plt.title(f'lateral vs mesial avg channels full profile')
    plt.xlim(-0.8, len(x_axis) - 1)
    for i in range(0, len(x_axis) - 1, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend(loc='lower right')
    plt.ylabel('% change in spike rate')
    plt.xticks(list(range(0, len(x_axis))))
    plt.locator_params(axis='y', nbins=15)
    plt.savefig(f'results/tempral_vs_mesial_avg_chans_full_profile_5min_{norm}_norm')
    plt.clf()


def all_subjects_baseline_vs_end(subjects, control=False, norm='amit', end='end', box=False):
    all_avg = []
    for subj in subjects.keys():
        channels = subjects[subj]
        blocks = []
        for chan in channels:
            chan_rates = pd.read_csv(f'results\\{subj}_{chan}_rates.csv' if not control else f'results\\control\\{subj}_{chan}_rates.csv', index_col=0)
            if end == 'end':
                y_axis = [chan_rates['rate'][0],
                          chan_rates['rate'][len(chan_rates) - 1]]
            else:
                y_axis = [chan_rates['rate'][0],
                          chan_rates['rate'][end]]
            if norm == 'baseline':
                y_axis = np.array(y_axis) / y_axis[0]
                y_axis = y_axis * 100 - 100
            elif norm == 'amit':
                y_axis = [0] + [((y_axis[1] - y_axis[0]) / max(y_axis[0], y_axis[1])) * 100]
            blocks.append(y_axis)

        avg_df = pd.DataFrame(blocks, columns=['baseline', 'end'])
        means = [avg_df[i].mean() for i in ['baseline', 'end']]
        all_avg.append(means)
        if not box:
            plt.plot(['baseline', 'end'], means, '-o')

    if norm is not None:
        plt.axhline(y=0, color='black', linestyle='--')
    if box:
        plt.boxplot([[x[0] for x in all_avg], [x[1] for x in all_avg]], showmeans=True)
        plt.xticks([1, 2], ['base', 'end'])
    else:
        all_avg_df = pd.DataFrame(all_avg, columns=['baseline', 'end'])
        all_means = [all_avg_df[i].mean() for i in ['baseline', 'end']]
        rect = plt.bar(['baseline', 'end'], all_means, alpha=0.5)
        plt.bar_label(rect, padding=1)
    plt.title(f'baseline vs end rate- top channels avg, {len(subjects)} subj')
    plt.xlabel('Time point')
    plt.ylabel('% Change in spike rate')
    counter1 = len([x for x in all_avg if x[1] - x[0] > 0])
    # plt.text(0, plt.ylim()[1] - 10, f'increase: {counter1}')


def all_subjects_baseline_avg(subjects, col='baseline'):
    all_subj = []
    for subj in subjects.keys():
        subj_avg = []
        df = pd.read_csv(f'results\\{subj}_chan_sum.csv')
        for chan in subjects[subj]:
            subj_avg.append(df[df['channel'] == chan][col].values[0])

        all_subj.append(np.array(subj_avg).mean())

    print(np.array(all_subj).mean())
    print(1)


frontal_stim = ['485', '486', '487', '488', '497', '498', '499', '505', '510-1', '510-7', '515', '520', '545']
# temporal_stim = ['489', '490', '496', '538']
mixed_stim = ['485', '497', '499', '505', '510-1', '510-7']
all_except_490_520 = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '538', '541', '544', '545']
manual_select = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '487': ['LAH2', 'LA2', 'LEC1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'], '515': ['RA1', 'RAH2'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
# round 4
# top_chan_profile_avg(manual_select, chan_num=1, norm='baseline', plot_subj=True)
# top_chans_profile_separate(manual_select)
# top_chan_profile_avg(manual_select, norm='baseline')
# top_chan_minute_profile_avg(manual_select)
# top_chan_block_stim_type(manual_select, error=True)
# top_chan_block_profile_avg_hemisphere(manual_select)
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

# all_subjects_baseline_vs_end(manual_select, norm=False, box=True)
# all_subjects_baseline_vs_end(manual_select, norm='baseline')
# all_subjects_baseline_vs_end(manual_select)
# all_subjects_baseline_avg(manual_select)
# all_subjects_baseline_avg(manual_select, col='sum')

manual_lateral = {'485': ['RPHG6'], '486': ['RPSMA5'], '488': ['LA7'],
                 '497': ['REC6', 'RPHG5', 'LAH6', 'LA5', 'LPHG5', 'RMH7'],
                 '499': ['LMH5'], '505': ['LEC5']}
manual_mesial = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '487': ['LAH2', 'LA2', 'LEC1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG1'],
                 '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'], '515': ['RA1', 'RAH2'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
# chans_profile_separate(manual_lateral, filename='lateral')
# lateral_vs_mesial(manual_lateral, manual_mesial)
# all_subjects_baseline_avg(manual_lateral)
# all_subjects_baseline_avg(manual_lateral, col='sum')
# stim_side_right = ['485', '487', '489', '490', '496', '497', '510-1', '510-7', '515', '520', '538', '544', '545']
#
# # only ipsi hemisphere
# manual_ipsi = {}
# for subj in manual_select.keys():
#     curr_side = 'R' if subj in stim_side_right else 'L'
#     top_chans = [x for x in manual_select[subj] if x[0] == curr_side]
#     if len(top_chans) > 0:
#         manual_ipsi[subj] = top_chans
#
# all_subjects_baseline_vs_end(manual_ipsi, norm='baseline')
#
# # top_chan_profile_avg(manual_ipsi, norm='baseline')
# # top_chan_minute_profile_avg(manual_ipsi)
# # top_chan_block_stim_type(manual_ipsi, error=True)
#
# lateral_ipsi = {}
# for subj in manual_lateral.keys():
#     curr_side = 'R' if subj in stim_side_right else 'L'
#     top_chans = [x for x in manual_lateral[subj] if x[0] == curr_side]
#     if len(top_chans) > 0:
#         lateral_ipsi[subj] = top_chans
# mesial_ipsi = {}
# for subj in manual_mesial.keys():
#     curr_side = 'R' if subj in stim_side_right else 'L'
#     top_chans = [x for x in manual_mesial[subj] if x[0] == curr_side]
#     if len(top_chans) > 0:
#         mesial_ipsi[subj] = top_chans
# lateral_vs_mesial(lateral_ipsi, mesial_ipsi)

control_chans = {'396': ['LAH1', 'LPHG2', 'LMH1', 'LEC1', 'LA2'], '398': ['LA1', 'LAH1', 'RAH1', 'RA2', 'REC3', 'LPHG1'],
                 '402': ['LA1', 'RAH1', 'LAH1', 'RA2', 'REC1'], '406': ['LPHG2', 'RPHG1', 'RAH1', 'RA2', 'LAH1', 'RMPF3'],
                 '415': ['LAH1', 'RA1', 'RAH1', 'LEC1', 'REC1'], '416': ['LAH2', 'RA2', 'RAH1', 'RIFGA1']}

# all_subjects_baseline_vs_end(control_chans, control=True)
# all_subjects_baseline_vs_end(manual_select)
# TODO: same funct but sum 1 first hour of combined nrem, then 2, then 3, then 4, then 5 and plot
chans_profile_separate(manual_select, csv_file='results\\%s_%s_nrem_rates.csv')
