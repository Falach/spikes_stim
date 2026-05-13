import mne
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import os
import matplotlib.patches as mpatches
import yasa

sr = 1000
edf_path = 'D:\\Maya\\p%s\\P%s_fixed.edf'
control_path = 'C:\\UCLA\\P%s_overnightData.edf'
edf_bipolar_path = 'D:\\Maya\\p%s\\P%s_bipolar.edf'
stim_path = 'D:\\Maya\\p%s\\p%s_stim_timing.csv'
scoring_path = 'D:\\Maya\\p%s\\p%s_sleep_scoring.m'
control_scoring_path = 'D:\\Ofer_backup\\V6_model\\hypnograms\\p%s_hypno.txt'
montage_path = 'D:\\Maya\\p%s\\MacroMontage.mat'
all_subj = ['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '515', '520',
            '538', '541', '544', '545']
all_control = ['396', '398', '402', '404', '405', '406', '415', '416']
control_stim = {'396': '485', '398': '486', '406': '488', '415': '489', '416': '496'}


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

    return stim_sessions


def get_stim_starts_upgraded(subj):
    hypno_path = rf'D:\Ofer_backup\V6_model\hypnograms\p{subj}_hypno.txt'
    # --- אפשרות א': קיים קובץ גירויים CSV (הלוגיקה המקורית שלך) ---
    if os.path.exists(stim_path % (subj, subj)):
        stim = np.array(pd.read_csv(stim_path % (subj, subj), header=None).iloc[0, :])
        stim_sessions = []
        start = stim[0] / 1000
        end = None
        for (i, x) in enumerate(stim):
            if end is not None:
                start = stim[i] / 1000
                end = None
            if i + 1 < stim.size and stim[i + 1] - stim[i] > 5 * 60 * 1000:
                end = stim[i] / 1000
                if stim[i] / 1000 - start > 60:
                    stim_sessions.append((start, end))
        return stim_sessions

    # --- אפשרות ב': קונטרול- אין קובץ גירויים, יצירת פרוטוקול לפי היפנוגרמה ---
    elif os.path.exists(hypno_path):
        # טעינת ההיפנוגרמה (קצב של 1Hz - כל ערך הוא שנייה)
        hypno = np.loadtxt(hypno_path)

        # 1. מציאת ה-NREM2 הראשון (ערך 2)
        nrem2_indices = np.where(hypno == 2)[0]
        if len(nrem2_indices) == 0:
            return []

        first_nrem2_time = nrem2_indices[0]

        # 2. המתנה של שעה (3600 שניות)
        search_start_time = first_nrem2_time + 3600

        # 3. הגדרת פרמטרים לחלון הבדיקה
        window_size = 80 * 60  # 80 דקות בשניות (4800 שניות)
        nrem_stages = [1, 2, 3]  # שלבי NREM

        # חיפוש הנקודה הראשונה שמתחילה ב-NREM2 ומקיימת 60% NREM ב-80 הדקות הבאות
        protocol_start_time = None
        for t in range(search_start_time, len(hypno) - window_size):
            if hypno[t] == 2:
                window = hypno[t: t + window_size]
                nrem_ratio = np.mean(np.isin(window, nrem_stages))

                if nrem_ratio >= 0.60:
                    protocol_start_time = t
                    break

        if protocol_start_time is None:
            return []

        # 4. יצירת רשימת טאפלים: 5 דקות גירוי, 5 דקות הפסקה לאורך 80 דקות
        stim_sessions = []
        stim_duration = 5 * 60  # 300 שניות
        pause_duration = 5 * 60  # 300 שניות
        cycle_duration = stim_duration + pause_duration  # 10 דקות סה"כ למחזור

        # מספר המחזורים שנכנסים ב-80 דקות (8 מחזורים)
        num_cycles = window_size // cycle_duration

        for c in range(num_cycles):
            curr_start = protocol_start_time + (c * cycle_duration)
            curr_end = curr_start + stim_duration
            stim_sessions.append((float(curr_start), float(curr_end)))

        return stim_sessions

    else:
        print(f"No data found for subject {subj}")
        return []

# get_stim_starts_upgraded('405')

# TODO: 497 check padding- first and last sec of block
def remove_stim(subj, block_raw, start, end):
    # stim = np.array(pd.read_csv(stim_path % (subj, subj), header=None).iloc[0, :])
    raws = []
    stim = pd.read_csv(stim_path % (subj, subj), header=None).iloc[0, :].to_list()
    stim = [round(x) for x in stim]
    # get the relevant section from the list with all stimuli
    start_stim = [x for x in stim if x >= start * 1000]
    end_stim = [x for x in stim if x <= end * 1000]
    if start_stim == [] or end_stim == [] or start_stim[0] >= end_stim[-1]:
        return block_raw
    else:
        start_stim = start_stim[0]
        end_stim = end_stim[-1]
    current = stim[len(stim) - stim[::-1].index(start_stim) - 1: stim.index(end_stim) + 1]
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

    return mne.concatenate_raws(raws)


def get_noisy_channels(subj, bi=False):
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
                if subj == '485' and chan['Area'][0] == 'RPHG':
                    continue
                to_remove.append(str(chan['Area'][0]) + str(chan_index))
            chan_index += 1
    if bi:
        bi_remove = [x[:-1]+str(int(x[-1]) - 1) for x in to_remove if x[-1].isnumeric()]
        to_remove.extend(bi_remove)
    return list(set(to_remove))

def get_control_clean_channels(subj, raw):
    to_remove = ['C3', 'C4', 'PZ', 'CZ', 'EZ', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'A1', 'A2', 'X1', 'X1-REF', 'ECG', 'EKG']
    if subj == '405':
        to_remove.extend(['RAH1', 'RAH6'])
    final = [chan for chan in raw.ch_names if chan.upper() not in to_remove]
    return final


def get_nrem_epochs(subj='510-1', with_stim=True):
    f = sio.loadmat(scoring_path % (subj, subj))
    data = f['sleep_score']
    scoring = np.array(data)[0]
    nrem = np.where(scoring == 1)
    nrem_epochs = []
    start = nrem[0][0]
    for i in range(len(nrem[0]) - 1):
        if nrem[0][i + 1] - nrem[0][i] != 1:
            nrem_epochs.append([start / 1000, nrem[0][i] / 1000])
            start = nrem[0][i + 1]

    # in case there is only one epoch
    # if nrem_epochs == []:
    nrem_epochs.append([start / 1000, nrem[0][-1] / 1000])

    if with_stim:
        nrem_stim_epochs = []
        stim = get_stim_starts(subj)
        for epoch in nrem_epochs[:]:
            for stim_start in stim:
                if epoch[0] <= stim_start[0] <= epoch[1]:
                    nrem_stim_epochs.append(epoch)
                    break

    return nrem_epochs, nrem_stim_epochs, stim

def is_stim_in_epoch(subj, start, end):
    stim = get_stim_starts(subj)
    for stim_start in stim:
        if start <= stim_start[0] <= end:
            return True

    return False

def get_control_nrem_epochs(subj='404'):
    scoring = np.loadtxt(control_scoring_path % subj)
    # n1 =1, n2 =2, n3 =3
    nrem = np.where(np.logical_and(scoring >= 2, scoring <= 3))
    nrem_epochs = []
    start = nrem[0][0]
    for i in range(len(nrem[0]) - 1):
        if nrem[0][i + 1] - nrem[0][i] != 1 and start != nrem[0][i]:
            # each epoch is 30 sec
            nrem_epochs.append([start, nrem[0][i]])
            start = nrem[0][i + 1]

    # in case the recording ends with nrem
    if nrem[0][-1] == len(scoring) - 1:
        nrem_epochs.append([start, nrem[0][-1]])

    return nrem_epochs

def get_stim_count(subj):
    stim = np.array(pd.read_csv(stim_path % (subj, subj), header=None).iloc[0, :])
    return len(stim)

# for choosing the most active channels
def calc_nrem_rate_per_chan(subjects, control=False):
    stim_val = {'before': 0, 'during': 1, 'after': 2}
    for subj in subjects:
        subj_files_list = glob.glob(f'results\\{subj}*split*' if not control else f'results\\control\\{subj}*split*')
        rates_per_chan = {'channel': [], 'before': [], 'during': [], 'after': [], 'sum': [], 'duration_sec': [], 'total_mean': []}
        for i, curr_file in enumerate(subj_files_list):
            if 'stim' not in curr_file:
                ch_name = curr_file.split(f'{subj}_')[1].split('_nrem')[0]
                chan_rates = pd.read_csv(curr_file, index_col=0)
                rates_per_chan['channel'].append(ch_name)
                for i in stim_val.keys():
                    curr_rates = chan_rates[chan_rates.is_stim == stim_val[i]]
                    total_duration = curr_rates['duration_sec'].sum() / 60
                    total_spikes = curr_rates['n_spikes'].sum()
                    if total_duration == 0:
                        rates_per_chan[i].append(None)
                    else:
                        rates_per_chan[i].append(total_spikes / total_duration)
                rates_per_chan['sum'].append(int(chan_rates['n_spikes'].sum()))
                rates_per_chan['duration_sec'].append(int(chan_rates['duration_sec'].sum()))
                rates_per_chan['total_mean'].append(rates_per_chan['sum'][-1] / (rates_per_chan['duration_sec'][-1] / 60))

        df = pd.DataFrame(rates_per_chan)
        df.to_csv(f'results\\{subj}_nrem_chan_sum.csv')


def sum_all_chans(subjects=['485'], path='results\\%s*split*'):
    for subj in subjects:
        subj_files_list = glob.glob(path % subj)
        rates = {'n_spikes': [], 'duration_sec': [], 'is_stim': []}
        for i, curr_file in enumerate(subj_files_list):
            chan_rates = pd.read_csv(curr_file, index_col=0)
            if i == 0:
                rates['n_spikes'] = np.zeros(len(chan_rates))
                rates['duration_sec'] = chan_rates['duration_sec'].tolist()
                rates['is_stim'] = chan_rates['is_stim'].tolist()

            rates['n_spikes'] += np.array(chan_rates['n_spikes'])


        df = pd.DataFrame(rates)
        df['rate'] = df['n_spikes'] / (df['duration_sec'] / 60)
        df.to_csv(f'results\\{subj}_nrem_all_sum.csv')

def get_top_chans(path, chan_num=5):
    df = pd.read_csv(path)
    top_chans = df.sort_values(by='before', ascending=False)['channel'].tolist()
    chans = []
    i = 0
    while len(chans) < chan_num and i < len(top_chans):
        if top_chans[i][:-1] not in [x[:-1] for x in chans]:
            chans.append(top_chans[i])
        i += 1

    return chans

def remove_noisy_samples_and_channels(
    subject_id,
    df_features,
    annotations_path,
    ts_col="onset_ms",
    channel_col="channel",
    bad_channels_dir=r"D:\Ofer_backup\bad_channels",
    print_stats=False
):
    df = df_features.copy()
    n_before = len(df)

    # מסכה מצטברת של דגימות רועשות
    noisy_mask = np.zeros(len(df), dtype=bool)

    if annotations_path is not None and os.path.exists(annotations_path):
        # ------------------------------------------------
        # 1. טעינת אנוטציות (CSV או FIF)
        # ------------------------------------------------
        if annotations_path.lower().endswith('.fif'):
            # טעינת האנוטציות מקובץ FIF (טוען רק metadata, לא את כל הסיגנל)
            raw_info = mne.io.read_raw_fif(annotations_path, preload=False)

            # ב-MNE ה-onset הוא בשניות, נמיר למילישניות
            ann = pd.DataFrame({
                "onset": raw_info.annotations.onset * 1000.0,  # המרה למילישניות
                "duration": raw_info.annotations.duration,  # בשניות
                "description": raw_info.annotations.description
            })
        else:
            # קריאה רגילה מ-CSV
            ann = pd.read_csv(annotations_path)
            ann["onset"] = ann["onset"].astype(float)  # הנחה שב-CSV זה כבר במילישניות לפי הקוד המקורי

        # חישוב התחלה וסוף במילישניות
        ann["start_ms"] = ann["onset"]
        ann["end_ms"] = ann["start_ms"] + ann["duration"].astype(float) * 1000.0

        channels = df[channel_col].astype(str).unique()

        # ------------------------------------------------
        # 2. סינון לפי האנוטציות
        # ------------------------------------------------
        for _, row in ann.iterrows():
            start = row["start_ms"]
            end = row["end_ms"]
            descr = str(row["description"])

            # אם התיאור הוא שם ערוץ - מסננים נקודתית
            if descr in channels:
                mask = (
                        (df[channel_col].astype(str) == descr) &
                        (df[ts_col] >= start) &
                        (df[ts_col] < end)
                )
            else:
                # סינון גורף לכל הערוצים (למשל עבור "BAD_ACQ" או "artifact")
                mask = (
                        (df[ts_col] >= start) &
                        (df[ts_col] < end)
                )

            noisy_mask |= mask
    else:
        if annotations_path is not None:
            print(f"Warning: annotations file not found: {annotations_path} (keeping all samples w.r.t annotations)")

    # ------------------------------------------------
    # 2. סינון לפי קובץ ערוצים רועשים (bad channels)
    # ------------------------------------------------
    if bad_channels_dir is not None and subject_id is not None:
        bad_channels_path = os.path.join(
            bad_channels_dir,
            f"p{subject_id}_bad_channels.txt"
        )

        if os.path.exists(bad_channels_path):
            with open(bad_channels_path, "r", encoding="utf-8") as f:
                bad_channels = {
                    line.strip()
                    for line in f
                    if line.strip()
                }

            if len(bad_channels) > 0:
                bad_mask = df[channel_col].astype(str).isin(bad_channels)
                noisy_mask |= bad_mask
        else:
            print(f"Warning: bad-channels file not found: {bad_channels_path} (keeping all channels)")

        # ------------------------------------------------
        # 3. סינון לפי זמני גירוי (Stimulation)
        # ------------------------------------------------
        if stim_path is not None:
            # הנחה: stim_path מכיל את ID הנבדק (e.g., stim_path % subj)
            full_stim_path = stim_path % (subject_id, subject_id)

            if os.path.exists(full_stim_path):
                # קריאת זמני הגירוי (במילישניות, כפי שמשתמע מהקוד המקורי)
                stim_times_ms = (
                    pd.read_csv(full_stim_path, header=None)
                    .iloc[0, :]
                    .dropna()
                    .astype(float)
                    .to_numpy()
                )

                if len(stim_times_ms) > 0:
                    # יצירת מסיכת סינון חדשה עבור כל זמני הגירוי
                    stim_mask = np.zeros(len(df), dtype=bool)
                    df_ts_ms = df[ts_col].to_numpy()  # זמני הספייקים במילישניות

                    for stim_time in stim_times_ms:
                        start_remove = stim_time - 500
                        end_remove = stim_time + 500

                        # יצירת מסכה עבור טווח גירוי בודד
                        mask_single_stim = (df_ts_ms >= start_remove) & (df_ts_ms < end_remove)
                        stim_mask |= mask_single_stim

                    # הוספת המקטעים שהוסרו בגלל הגירוי למסכה הכוללת
                    noisy_mask |= stim_mask
            else:
                print(f"Warning: Stimulation file not found: {full_stim_path} (keeping all samples w.r.t stim)")

        # ------------------------------------------------
        # 4. החלת המסכה והדפסה
        # ------------------------------------------------
        cleaned = df.loc[~noisy_mask].copy()
        if print_stats:
            n_after = len(cleaned)
            n_removed = n_before - n_after

            if n_before > 0:
                pct_removed = 100.0 * n_removed / n_before
            else:
                pct_removed = 0.0

            print(
                f"Removed {n_removed} / {n_before} samples "
                f"({pct_removed:.1f}%) as noisy "
                f"(annotations + bad channels + stimulation)"
            )

        return cleaned


def plot_hypno_spectrogram_yasa(subj_list, hypno_path_template, raw_path_template, stim_func, channel='C3'):
    for subj in subj_list:
        hypno_path = hypno_path_template % subj
        raw_path = raw_path_template % subj

        if not os.path.exists(hypno_path) or not os.path.exists(raw_path):
            print(f"Missing files for {subj}, skipping...")
            continue

        # 1. טעינת נתונים
        hypno = np.loadtxt(hypno_path)
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)

        # בחירת ערוץ
        target_ch = [ch for ch in raw.ch_names if channel.upper() in ch.upper()]
        if not target_ch:
            print(f"Channel {channel} not found in {subj}")
            continue
        raw.pick_channels([target_ch[0]]).resample(100)  # דגימה מחדש ל-100Hz

        # fs = raw.info['sfreq']
        data = raw.get_data()[0] * 1e6  # המרה ל-uV (YASA אוהבת סדרי גודל כאלו)

        # 2. יצירת הגרף בעזרת YASA
        # הפונקציה מחזירה אובייקט Figure
        # converting 1Hz to fs Hz by repeating each value fs times
        hyp_fs = np.repeat(hypno, 100)
        # number of samples in EDF
        n_edf = len(data)
        # if hypnogram length too long trimming to EDF length
        if len(hyp_fs) > n_edf:
            hyp_fs = hyp_fs[:n_edf]
        # if hypnogram length too short complete to EDF length
        elif len(hyp_fs) < n_edf:
            hyp_fs = np.pad(hyp_fs, (0, n_edf - len(hyp_fs)), mode="edge")
        fig = yasa.plot_spectrogram(data, 100, hyp_fs, trimperc=2.5)

        # קבלת הצירים מתוך ה-Figure (בדרך כלל יש 3: היפנוגרמה, ספקטרוגרמה, ו-colorbar)
        axes = fig.get_axes()
        ax_hyp = axes[0]  # הציר העליון של ההיפנוגרמה

        # 3. הוספת זמני הגירוי על ציר ההיפנוגרמה
        stims = stim_func(subj)
        added_label = False
        for start, end in stims:
            # ב-YASA ציר ה-X הוא בשעות כברירת מחדל
            rect = mpatches.Rectangle((start / 3600, -0.5), (end - start) / 3600, 5,
                                      color='cyan', alpha=0.4,
                                      label='Stimulation' if not added_label else "")
            ax_hyp.add_patch(rect)
            added_label = True

        if added_label:
            ax_hyp.legend(loc='upper right', fontsize='small')

        # עדכון הכותרת
        ax_hyp.set_title(f"Subject {subj} - Sleep Architecture & Stimuli")

        # 4. שמירה
        save_name = rf"D:\Ofer_backup\V6_model\controls\yasa_summary_{subj}.png"
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_name}")

# plot_hypno_spectrogram_yasa(['394', '396', '398', '402', '404', '406', '415', '416'], r"D:\Ofer_backup\V6_model\hypnograms\p%s_hypno.txt",
#                            r"D:\UCLA\P%s_overnightData.edf", get_stim_starts_upgraded, channel='C3')

manual_select_14_thresh = {'485': ['RMH1', 'RPHG1', 'RBAA1'], '489': ['LPHG2', 'RAH1', 'RPHG1'],
                 '497': ['RPHG2', 'REC2', 'LAH1', 'LPHG3', 'RMH2', 'LEC1'], '498': ['REC2', 'RMH1', 'RA2', 'RPHG4'],
                 '499': ['LMH5'], '505': ['LEC1', 'LA2', 'LAH3'], '510-7': ['RAH1'],
                 '520': ['REC1', 'RMH1', 'LMH1'], '545': ['LAH3', 'REC1']}
# avg_block_size(all_subj)
# plot_stim_duration(all_subj)
# get_control_nrem_epochs()
# calc_nrem_rate_per_chan(['396', '398', '402', '404', '405', '406', '415', '416'], control=False)
# sum_all_chans(['485', '486', '487', '488', '489', '496', '497', '498', '499', '505', '510-1', '510-7', '520', '538', '541', '544'])
# sum_all_chans([x for x in all_control if x != '405'])

# noisy = {}
# for subj in all_subj:
#     current = get_noisy_channels(subj)
#     noisy[subj] = [x for x in current if x not in ['C3', 'C4', 'PZ', 'CZ', 'EZ', 'EMG1', 'EMG2', 'EOG1', 'EOG2', 'A1', 'A2']]
#     print(subj, noisy[subj])