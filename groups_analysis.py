import numpy as np
import pandas as pd
import utils
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# NUM_OF_FEATURES = 7
# TIMESTAMP_INDEX = 0
# CHANNEL_INDEX = 1
# AMPLITUDE_INDEX = 2
# DURATION_INDEX = 3
# CORD_X_INDEX = 4
# CORD_Y_INDEX = 5
# CORD_Z_INDEX = 6
#
# # Additional features - added late
# GROUP_INDEX = 7
# GROUP_FOCAL_INDEX = 8
# GROUP_EVENT_DURATION_INDEX = 9
# GROUP_EVENT_SIZE_INDEX = 10
# GROUP_EVENT_DEEPEST_INDEX = 11
# GROUP_EVENT_SHALLOWEST_INDEX = 12
# GROUP_EVENT_SPATIAL_SPREAD_INDEX = 13
# IS_IN_SCALP_INDEX = 14
# STIMULI_FLAG_INDEX = 15
# HYPNOGRAM_FLAG_INDEX = 16
# SUBJECT_NUMBER = 17
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
# spikes_path = r'C:\repos\NirsLabProject\NirsLabProject\data\products\p%s\bipolar_model\features\flat_features.npy'
# groups_path = r'C:\repos\NirsLabProject\NirsLabProject\data\products\p%s\bipolar_model\features\groups.npy'
spikes_path = r'D:\Ofer_backup\p%s\bipolar_model\features\flat_features.npy'
groups_path = r'D:\Ofer_backup\p%s\bipolar_model\features\groups.npy'
columns = ['index', 'size', 'timestamp', 'last_event_timestamp', 'focal_channel_name', 'hemispheres', 'structures',
           'deepest_electrode', 'shallowest_electrode', 'group_spatial_spread', 'amplitude', 'duration',
           'group_duration', 'stimuli', 'hypnogram', 'subject']
frontal_chans = [f"{string}{number}" for string in ['ROF', 'LOF', 'RAC', 'LAC', 'LAF', 'RAF', 'RFA', 'RIF-dAC', 'LOPR'] for number in range(1, 9)]
temporal_chans = [f"{string}{number}" for string in ['RA', 'LA', 'RAH', 'LAH', 'RMH', 'LMH', 'REC', 'LEC', 'RPHG', 'LPHG'] for number in range(1, 9)]
for_avg_rates = {'all': [], 'ipsi': [], 'contra': [], 'frontal': [], 'temporal': []}
for_avg_props = {'amp': [], 'size': [], 'duration': [], 'spread': []}
plot_rates = True
scaler = StandardScaler()
for subj in [x for x in subjects if x not in ['515', '541', '545']]:
# for subj in ['541']:
    stim_side = 'R' if subj in stim_side_right else 'L'
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
    # final_features = merged_df[~merged_df['focal_channel_name'].isin(utils.get_noisy_channels(subj, True))]

    nrem_epochs, nrem_stim_epochs, stim_epochs = utils.get_nrem_epochs(subj)
    # take epochs after sleep onset but before first stimuli as baseline
    baseline = final_features[(final_features['timestamp'] > nrem_epochs[0][0] * 1000) & (final_features['timestamp'] < stim_epochs[0][0] * 1000)]
    baseline_minutes = (stim_epochs[0][0] - nrem_epochs[0][0]) / 60

    # for debug
    # raw = mne.io.read_raw_edf(utils.edf_path % (subj, subj), preload=True).crop(tmin=0, tmax=nrem_epochs[1][0])
    # raw.set_annotations(mne.Annotations(list(baseline['timestamp']/1000), [0.1] * len(baseline), ['spike'] * len(baseline)))
    # raw.plot(duration=30)
    # end debug

    base_rate = len(baseline) / baseline_minutes
    rates = {'all': [base_rate],
             'ipsi': [len(baseline[baseline['focal_channel_name'].str[0] == stim_side]) / baseline_minutes],
             'contra': [len(baseline[baseline['focal_channel_name'].str[0] != stim_side]) / baseline_minutes],
             'frontal': [len(baseline[baseline['focal_channel_name'].isin(frontal_chans)]) / baseline_minutes],
             'temporal': [len(baseline[baseline['focal_channel_name'].isin(temporal_chans)]) / baseline_minutes]}
    props = {'amp': [np.mean(baseline['amplitude'])],
             'size': [np.mean(baseline['size'])],
             'duration': [np.mean(baseline['duration'])],
             'spread': [np.mean(baseline['group_spatial_spread'])]}
    n_chans = {'all': final_features['focal_channel_name'].nunique(),
               'ipsi': final_features[final_features['focal_channel_name'].str[0] == stim_side]['focal_channel_name'].nunique(),
               'contra': final_features[final_features['focal_channel_name'].str[0] != stim_side]['focal_channel_name'].nunique(),
               'frontal': final_features[final_features['focal_channel_name'].isin(frontal_chans)]['focal_channel_name'].nunique(),
               'temporal': final_features[final_features['focal_channel_name'].isin(temporal_chans)]['focal_channel_name'].nunique()}
    # Append values using list comprehensions
    # my_dict = {key: arr + [value] for key, arr, value in zip(my_dict.keys(), my_dict.values(), values_to_append)}
    for i, (start, end) in enumerate(stim_epochs):
        stim_block = final_features[(final_features['timestamp'] > start * 1000) & (final_features['timestamp'] < end * 1000)]
        stim_minutes = (end - start) / 60
        rates['all'].append(len(stim_block) / stim_minutes)
        rates['ipsi'].append(len(stim_block[stim_block['focal_channel_name'].str[0] == stim_side]) / stim_minutes)
        rates['contra'].append(len(stim_block[stim_block['focal_channel_name'].str[0] != stim_side]) / stim_minutes)
        rates['frontal'].append(len(stim_block[stim_block['focal_channel_name'].isin(frontal_chans)]) / stim_minutes)
        rates['temporal'].append(len(stim_block[stim_block['focal_channel_name'].isin(temporal_chans)]) / stim_minutes)
        props['amp'].append(np.mean(stim_block['amplitude']))
        props['size'].append(np.mean(stim_block['size']))
        props['duration'].append(np.mean(stim_block['duration']))
        props['spread'].append(np.mean(stim_block['group_spatial_spread']))
        if i + 1 < len(stim_epochs):
            pause_block = final_features[(final_features['timestamp'] > end * 1000) & (final_features['timestamp'] < stim_epochs[i + 1][0] * 1000)]
            pause_minutes = (stim_epochs[i + 1][0] - end) / 60
            rates['all'].append(len(pause_block) / pause_minutes)
            rates['ipsi'].append(len(pause_block[pause_block['focal_channel_name'].str[0] == stim_side]) / pause_minutes)
            rates['contra'].append(len(pause_block[pause_block['focal_channel_name'].str[0] != stim_side]) / pause_minutes)
            rates['frontal'].append(
                len(pause_block[pause_block['focal_channel_name'].isin(frontal_chans)]) / stim_minutes)
            rates['temporal'].append(
                len(pause_block[pause_block['focal_channel_name'].isin(temporal_chans)]) / stim_minutes)
            props['amp'].append(np.mean(pause_block['amplitude']))
            props['size'].append(np.mean(pause_block['size']))
            props['duration'].append(np.mean(pause_block['duration']))
            props['spread'].append(np.mean(pause_block['group_spatial_spread']))
        else:
            # for now using only 5 minutes after last stimuli
            after = final_features[(final_features['timestamp'] > end * 1000) & (final_features['timestamp'] < (end + 60 * 5) * 1000)]
            rates['all'].append(len(after) / 5)
            rates['ipsi'].append(len(after[after['focal_channel_name'].str[0] == stim_side]) / 5)
            rates['contra'].append(len(after[after['focal_channel_name'].str[0] != stim_side]) / 5)
            rates['frontal'].append(len(after[after['focal_channel_name'].isin(frontal_chans)]) / stim_minutes)
            rates['temporal'].append(len(after[after['focal_channel_name'].isin(temporal_chans)]) / stim_minutes)
            props['amp'].append(np.mean(after['amplitude']))
            props['size'].append(np.mean(after['size']))
            props['duration'].append(np.mean(after['duration']))
            props['spread'].append(np.mean(after['group_spatial_spread']))

# plot the spike rate in each epoch here
    if plot_rates:
        # for (curr_rates, legend) in zip((rates, rates_ipsi, rates_contra), ('all', 'ipsi', 'contra')):
        for (legend, curr_rates) in rates.items():
            norm = scaler.fit_transform(np.array(curr_rates).reshape(-1,1)).flatten()
            # norm = [(x - curr_rates[0]) / max(x, curr_rates[0]) for x in curr_rates]
            for_avg_rates[legend].append(norm)
            plt.plot(norm, '-o', label=legend + ' ' + str(n_chans[legend]))
    else:
        for (legend, curr_rates) in props.items():
            # norm = [(x - curr_rates[0]) / max(x, curr_rates[0]) for x in curr_rates]
            norm = scaler.fit_transform(np.array(curr_rates).reshape(-1,1)).flatten()
            for_avg_props[legend].append(norm)
            plt.plot(norm, '-o', label=legend)
    # plt.axhline(y=0, color='red', linestyle='dashed')
    for i in range(0, len(rates['all']) - 1, 2):
        plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
    sleep_percent = pd.read_csv(f'results/{subj}_sleep_percent.csv')
    for k, j in enumerate(sleep_percent['sleep_percent']):
        plt.text(k - 0.7, plt.ylim()[1] -0.05, str(int(j)), size=8)
    plt.title(f'{subj} - groups profile')
    plt.xlabel('Time point')
    plt.ylabel('z-score')
    plt.legend(fontsize='small')
    plt.xlim(0, len(rates['all']) - 1)
    plt.tight_layout()
    # plt.savefig(f'results/groups/{subj}_groups_rates_z.png')
    plt.clf()

if plot_rates:
    max_len = max(map(len, for_avg_rates['all']))
    # plot the avg rate in each epoch here
    for (legend, curr_rates) in for_avg_rates.items():
        padded_lists = np.array([list(lst) + [np.nan] * (max_len - len(lst)) for lst in curr_rates])
        avg_rates = np.nanmean(padded_lists, axis=0)
        std_rates = np.nanstd(padded_lists, axis=0)
        plt.plot(avg_rates, '-o', label=legend)
        # plt.fill_between(range(len(avg_rates)), avg_rates - std_rates, avg_rates + std_rates, alpha=0.2)
else:
    max_len = max(map(len, for_avg_props['amp']))
    for (legend, curr_rates) in for_avg_props.items():
        padded_lists = np.array([list(lst) + [np.nan] * (max_len - len(lst)) for lst in curr_rates])
        avg_rates = np.nanmean(padded_lists, axis=0)
        std_rates = np.nanstd(padded_lists, axis=0)
        plt.plot(avg_rates, '-o', label=legend)
        # plt.fill_between(range(len(avg_rates)), avg_rates - std_rates, avg_rates + std_rates, alpha=0.2)
# plt.axhline(y=0, color='red', linestyle='dashed')
for i in range(0, max_len - 1, 2):
    plt.axvspan(i, i + 1, facecolor='silver', alpha=0.5)
plt.title(f'All subjects groups profile')
plt.xlabel('Time point')
plt.ylabel('z-score')
plt.legend(loc='lower left')
plt.xlim(0, max_len - 1)
plt.tight_layout()
plt.savefig(f'results/groups/all_groups_rates_z.png')
print(1)



