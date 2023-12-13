import numpy as np
import pandas as pd
import utils
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#
# STIMULI_FLAG_BEFORE_FIRST_STIMULI_SESSION = 0
# STIMULI_FLAG_STIMULI_SESSION = 1  # during stimuli session that have multiple windows
# STIMULI_FLAG_DURING_STIMULI_BLOCK = 2  # during a stimuli window
# STIMULI_FLAG_AFTER_STIMULI_SESSION = 3
#
# HYPNOGRAM_FLAG_REM_OR_WAKE = 0
# HYPNOGRAM_FLAG_NREM = 1


subjects = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '520', '538',
            '541', '544', '545']
controls = ['396', '398', '402', '406']
stim_side_right = ['485', '487', '489', '490', '496', '497', '510-1', '510-7', '515', '520', '538', '544', '545',
                       '396', '398', '404', '406', '415']
spikes_path = r'D:\Ofer_backup\p%s\bipolar_model\features\flat_features.npy'
groups_path = r'D:\Ofer_backup\p%s\bipolar_model\features\groups.npy'
columns = ['index', 'size', 'timestamp', 'last_event_timestamp', 'focal_channel_name', 'hemispheres', 'structures',
           'deepest_electrode', 'shallowest_electrode', 'group_spatial_spread', 'amplitude', 'duration',
           'group_duration', 'stimuli', 'hypnogram', 'subject']

frontal_chans = [f"{string}{number}" for string in ['ROF', 'LOF', 'RAC', 'LAC', 'LAF', 'RAF', 'RFA', 'RIF-dAC', 'LOPR'] for number in range(1, 9)]
temporal_chans = [f"{string}{number}" for string in ['RA', 'LA', 'RAH', 'LAH', 'RMH', 'LMH', 'REC', 'LEC', 'RPHG', 'LPHG'] for number in range(1, 9)]
for_avg_props = {}
plot_rates = True
scaler = StandardScaler()

def get_subj_data(subj):
    group_features = np.load(groups_path % subj, allow_pickle=True)
    group_features_df = pd.DataFrame(list(group_features))
    spikes_features = np.load(spikes_path % subj, allow_pickle=True)
    spikes_features_df = pd.DataFrame(spikes_features,
                                      columns=['timestamp', 'channel', 'amplitude', 'duration', 'x', 'y', 'z', 'group',
                                               'group_focal', 'group_duration', 'group_size', 'group_depth',
                                               'group_shallow', 'group_spread', 'is_scalp', 'stimuli', 'hypnogram',
                                               'subject'])
    merged_df = pd.merge(group_features_df, spikes_features_df, how='inner', left_on=['index'], right_on=['group'])
    unique_indices = np.unique(merged_df['group'], return_index=True)[1]
    clean_df = merged_df.loc[unique_indices, columns]
    final_features = clean_df[~clean_df['focal_channel_name'].isin(utils.get_noisy_channels(subj, True))]

    return final_features

def init_dicts_with_baseline_rates(subj_features, nrem_start, stim_start, side):
    # take epochs after sleep onset but before first stimuli as baseline
    baseline = subj_features[(subj_features['timestamp'] > nrem_start * 1000) &
                             (subj_features['timestamp'] < stim_start * 1000)]
    baseline_minutes = (stim_start - nrem_start) / 60
    base_rate = len(baseline) / baseline_minutes
    rates = {'all': [base_rate],
             'ipsi': [len(baseline[baseline['focal_channel_name'].str[0] == side]) / baseline_minutes],
             'contra': [len(baseline[baseline['focal_channel_name'].str[0] != side]) / baseline_minutes],
             'frontal': [len(baseline[baseline['focal_channel_name'].isin(frontal_chans)]) / baseline_minutes],
             'temporal': [len(baseline[baseline['focal_channel_name'].isin(temporal_chans)]) / baseline_minutes]}
    props = {'rates': rates}
    for prop in ['amplitude', 'size', 'duration', 'group_duration', 'group_spatial_spread']:
        curr = {'all': [np.mean(baseline[prop])],
                'ipsi': [np.mean(baseline[baseline['focal_channel_name'].str[0] == side][prop])],
                'contra': [np.mean(baseline[baseline['focal_channel_name'].str[0] != side][prop])],
                'frontal': [np.mean(baseline[baseline['focal_channel_name'].isin(frontal_chans)][prop])],
                'temporal': [np.mean(baseline[baseline['focal_channel_name'].isin(temporal_chans)][prop])]}
        props[prop] = curr

    n_chans = {'all': subj_features['focal_channel_name'].nunique(),
               'ipsi': subj_features[subj_features['focal_channel_name'].str[0] == side][
                   'focal_channel_name'].nunique(),
               'contra': subj_features[subj_features['focal_channel_name'].str[0] != side][
                   'focal_channel_name'].nunique(),
               'frontal': subj_features[subj_features['focal_channel_name'].isin(frontal_chans)][
                   'focal_channel_name'].nunique(),
               'temporal': subj_features[subj_features['focal_channel_name'].isin(temporal_chans)][
                   'focal_channel_name'].nunique()}
    # for debug
    # raw = mne.io.read_raw_edf(utils.edf_path % (subj, subj), preload=True).crop(tmin=0, tmax=nrem_epochs[1][0])
    # raw.set_annotations(mne.Annotations(list(baseline['timestamp']/1000), [0.1] * len(baseline), ['spike'] * len(baseline)))
    # raw.plot(duration=30)
    # end debug
    return props, n_chans

def append_to_props(props, curr_df, side, time):
    # rates
    props['rates']['all'].append(len(curr_df) / stim_minutes)
    props['rates']['ipsi'].append(len(curr_df[curr_df['focal_channel_name'].str[0] == side]) / time)
    props['rates']['contra'].append(len(curr_df[curr_df['focal_channel_name'].str[0] != side]) / time)
    props['rates']['frontal'].append(len(curr_df[curr_df['focal_channel_name'].isin(frontal_chans)]) / time)
    props['rates']['temporal'].append(len(curr_df[curr_df['focal_channel_name'].isin(temporal_chans)]) / time)
    # props
    for prop in ['amplitude', 'size', 'duration', 'group_duration', 'group_spatial_spread']:
        props[prop]['all'].append(np.mean(curr_df[prop]))
        props[prop]['ipsi'].append(np.mean(curr_df[curr_df['focal_channel_name'].str[0] == side][prop]))
        props[prop]['contra'].append(np.mean(curr_df[curr_df['focal_channel_name'].str[0] != side][prop]))
        props[prop]['frontal'].append(np.mean(curr_df[curr_df['focal_channel_name'].isin(frontal_chans)][prop]))
        props[prop]['temporal'].append(np.mean(curr_df[curr_df['focal_channel_name'].isin(temporal_chans)][prop]))


def plot_layout(subj, prop_len, prop_name, base_relative=True, to_save=True):
    if base_relative:
        plt.axhline(y=0, color='red', linestyle='dashed')
    for i in range(0, prop_len - 1, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    if subj != 'all':
        sleep_percent = pd.read_csv(f'results/{subj}_sleep_percent.csv')
        for k, j in enumerate(sleep_percent['sleep_percent']):
            plt.text(k - 0.7, plt.ylim()[1] -0.05, str(int(j)), size=8)
    title = f'{subj} - {prop_name} groups profile'
    plt.title(title)
    plt.xlabel('Time point')
    plt.ylabel('% change from base' if base_relative else 'z-score')
    plt.legend(fontsize='small')
    plt.xlim(0, prop_len - 1)
    plt.tight_layout()
    if to_save:
        title += '_base' if base_relative else '_z'
        plt.savefig(f'results/groups/{title}.png')
    plt.clf()

for subj in [x for x in subjects if x not in ['515', '541', '545']]:
# for subj in ['541']:
    stim_side = 'R' if subj in stim_side_right else 'L'
    final_features = get_subj_data(subj)
    nrem_epochs, nrem_stim_epochs, stim_epochs = utils.get_nrem_epochs(subj)
    props, n_chans = init_dicts_with_baseline_rates(final_features, nrem_epochs[0][0], stim_epochs[0][0], stim_side)

    for i, (start, end) in enumerate(stim_epochs):
        stim_block = final_features[(final_features['timestamp'] > start * 1000) & (final_features['timestamp'] < end * 1000)]
        stim_minutes = (end - start) / 60
        append_to_props(props, stim_block, stim_side, stim_minutes)
        if i + 1 < len(stim_epochs):
            pause_block = final_features[(final_features['timestamp'] > end * 1000) & (final_features['timestamp'] < stim_epochs[i + 1][0] * 1000)]
            pause_minutes = (stim_epochs[i + 1][0] - end) / 60
            append_to_props(props, pause_block, stim_side, pause_minutes)
        else:
            # for now using only 5 minutes after last stimuli
            after = final_features[(final_features['timestamp'] > end * 1000) & (final_features['timestamp'] < (end + 60 * 5) * 1000)]
            append_to_props(props, after, stim_side, 5)

# plot the spike rate in each epoch here
#     if plot_rates:
    for prop in props.keys():
        if prop not in for_avg_props.keys():
            for_avg_props[prop] = {legend: [] for legend in props[prop].keys()}
        for (legend, curr_rates) in props[prop].items():
            norm = scaler.fit_transform(np.array(curr_rates).reshape(-1,1)).flatten()
            # norm = [(x - curr_rates[0]) / max(x, curr_rates[0]) for x in curr_rates]
            for_avg_props[prop][legend].append(norm)
            plt.plot(norm, '-o', label=legend + ' ' + str(n_chans[legend]))
        plot_layout(subj, len(props[prop]['all']), prop, False, False)


max_len = max(map(len, for_avg_props['rates']['all']))
# plot the avg rate in each epoch here
for prop in for_avg_props.keys():
    for (legend, curr_rates) in for_avg_props[prop].items():
        padded_lists = np.array([list(lst) + [np.nan] * (max_len - len(lst)) for lst in curr_rates])
        avg_rates = np.nanmean(padded_lists, axis=0)
        std_rates = np.nanstd(padded_lists, axis=0)
        plt.plot(avg_rates, '-o', label=legend)
        # plt.fill_between(range(len(avg_rates)), avg_rates - std_rates, avg_rates + std_rates, alpha=0.2)
    plot_layout('all', max_len, prop, False, True)

print(1)