import mne
import pandas as pd
import numpy as np
import scipy.stats as sp_stats
import scipy.signal as sp_sig
import antropy as ant
from scipy.integrate import simps
from pathlib import Path, PurePath
from matplotlib import pyplot as plt
from sklearn.preprocessing import robust_scale
import joblib
from utils import sr, get_control_clean_channels, control_path
from pyprep.prep_pipeline import PrepPipeline
import pyprep
# from detector_with_conj import detect

model_lgbm = joblib.load('models\LGBM_V1.pkl')
model_rf = joblib.load('models\RF_V1.pkl')


def bandpower_from_psd_ndarray(psd, freqs, bands, relative=True):
    # Type checks
    assert isinstance(bands, list), 'bands must be a list of tuple(s)'
    assert isinstance(relative, bool), 'relative must be a boolean'

    # Safety checks
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert freqs.ndim == 1, 'freqs must be a 1-D array of shape (n_freqs,)'
    assert psd.shape[-1] == freqs.shape[-1], 'n_freqs must be last axis of psd'

    # Extract frequencies of interest
    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]

    # Trim PSD to frequencies of interest
    psd = psd[..., idx_good_freq]

    # Check if there are negative values in PSD
    if (psd < 0).any():
        msg = (
            "There are negative values in PSD. This will result in incorrect "
            "bandpower values. We highly recommend working with an "
            "all-positive PSD. For more details, please refer to: "
            "https://github.com/raphaelvallat/yasa/issues/29")
        print(msg)

    # Calculate total power
    total_power = simps(psd, dx=res, axis=-1)
    total_power = total_power[np.newaxis, ...]

    # Initialize empty array
    bp = np.zeros((len(bands), *psd.shape[:-1]), dtype=np.float)

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[i] = simps(psd[..., idx_band], dx=res, axis=-1)

    if relative:
        bp /= total_power
    return bp


def calc_features(epochs, subj):
    # Bandpass filter
    freq_broad = (0.1, 500)
    # FFT & bandpower parameters
    sr = 1000
    bands = [
        (0.1, 4, 'delta'), (4, 8, 'theta'),
        (8, 12, 'alpha'), (12, 16, 'sigma'), (16, 30, 'beta'),
        (30, 100, 'gamma'), (100, 300, 'fast')
    ]

    # Calculate standard descriptive statistics
    hmob, hcomp = ant.hjorth_params(epochs, axis=1)

    feat = {
        'subj': np.full(len(epochs), subj),
        'epoch_id': np.arange(len(epochs)),
        'std': np.std(epochs, ddof=1, axis=1),
        'iqr': sp_stats.iqr(epochs, axis=1),
        'skew': sp_stats.skew(epochs, axis=1),
        'kurt': sp_stats.kurtosis(epochs, axis=1),
        'nzc': ant.num_zerocross(epochs, axis=1),
        'hmob': hmob,
        'hcomp': hcomp
    }

    # Calculate spectral power features (for EEG + EOG)
    freqs, psd = sp_sig.welch(epochs, sr)
    bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
    for j, (_, _, b) in enumerate(bands):
        feat[b] = bp[j]

    # Add power ratios for EEG
    delta = feat['delta']
    feat['dt'] = delta / feat['theta']
    feat['ds'] = delta / feat['sigma']
    feat['db'] = delta / feat['beta']
    feat['dg'] = delta / feat['gamma']
    feat['df'] = delta / feat['fast']
    feat['at'] = feat['alpha'] / feat['theta']
    feat['gt'] = feat['gamma'] / feat['theta']
    feat['ft'] = feat['fast'] / feat['theta']
    feat['ag'] = feat['gamma'] / feat['alpha']
    feat['af'] = feat['fast'] / feat['alpha']

    # Add total power
    idx_broad = np.logical_and(
        freqs >= freq_broad[0], freqs <= freq_broad[1])
    dx = freqs[1] - freqs[0]
    feat['abspow'] = np.trapz(psd[:, idx_broad], dx=dx)

    # Calculate entropy and fractal dimension features
    feat['perm'] = np.apply_along_axis(
        ant.perm_entropy, axis=1, arr=epochs, normalize=True)
    feat['higuchi'] = np.apply_along_axis(
        ant.higuchi_fd, axis=1, arr=epochs)
    feat['petrosian'] = ant.petrosian_fd(epochs, axis=1)

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    # feat.index.name = 'epoch'

    ############################
    # SMOOTHING & NORMALIZATION
    ############################
    roll1 = feat.rolling(window=1, center=True, min_periods=1, win_type='triang').mean()
    roll1[roll1.columns] = robust_scale(roll1, quantile_range=(5, 95))
    roll1 = roll1.iloc[:, 1:].add_suffix('_cmin_norm')

    roll3 = feat.rolling(window=3, center=True, min_periods=1, win_type='triang').mean()
    roll3[roll3.columns] = robust_scale(roll3, quantile_range=(5, 95))
    roll3 = roll3.iloc[:, 1:].add_suffix('_pmin_norm')

    # Add to current set of features
    feat = feat.join(roll1).join(roll3)

    return feat


def format_raw(raw):
    epochs = []
    window_size = int(sr / 4)
    raw.load_data()

    # Create bipolar channel
    raw_bi = mne.set_bipolar_reference(raw, raw.ch_names[0], raw.ch_names[1], ch_name='bi')
    raw_data = raw_bi.get_data()[0]

    # Normalization
    raw_data = sp_stats.zscore(raw_data)

    for i in range(0, len(raw_data), window_size):
        curr_block = raw_data[i: i + window_size]
        if i + window_size < len(raw_data):
            epochs.append(curr_block)

    return np.array(epochs)


def detect_spikes(raw, plot=False):
    x = format_raw(raw)
    features = calc_features(x, subj)
    # check nans
    # np.where(np.asanyarray(np.isnan(features[model_lgbm.feature_name_])))
    features = np.nan_to_num(features[model_lgbm.feature_name_])
    y_lgbm = model_lgbm.predict(features)
    y_rf = model_rf.predict(features)
    y = np.array(y_lgbm) + np.array(y_rf)
    y[y == 2] = 1
    if plot:
        spikes_onsets = np.where(y == 1)[0] / (raw.info['sfreq'] / 250)
        raw.set_annotations(mne.Annotations(spikes_onsets, [0.25] * len(spikes_onsets), ['spike'] * len(spikes_onsets)))
        mne.set_bipolar_reference(raw, raw.ch_names[0], raw.ch_names[1], ch_name='bi', drop_refs=False).plot(
            duration=30, scalings='auto')
        # raw.plot(duration=30)
    return y


def fill_row(raw, rates, is_stim=False):
    spikes = detect_spikes(raw)
    rates['n_spikes'].append(spikes.sum())
    rates['duration_sec'].append(raw.n_times / sr)
    rates['rate'].append(spikes.sum() / (raw.n_times / sr / 60))
    len_20_percent = int(len(spikes) / 5)
    duration_20_sec = len_20_percent / (sr / 250)
    rates['duration_20%'].append(duration_20_sec)
    for i in range(1, 6):
        n_20 = spikes[len_20_percent * (i - 1): len_20_percent * i].sum()
        rates[f'n_{str(i)}_20%'].append(n_20)
        rates[f'rate_{str(i)}_20%'].append(n_20 / (duration_20_sec / 60))

    return rates

def detect_subj_chan(subj, channels):
    rates = {'n_spikes': [], 'duration_sec': [], 'rate': [], 'duration_20%': [], 'n_1_20%': [], 'n_2_20%': [],
             'n_3_20%': [], 'n_4_20%': [], 'n_5_20%': [], 'rate_1_20%': [], 'rate_2_20%': [], 'rate_3_20%': [],
             'rate_4_20%': [], 'rate_5_20%': []}
    stim_sections_sec = [(subjects_nrem[subj] * 60, subjects_nrem[subj] * 60 + 60 * 60)]
    stim_start_sec = stim_sections_sec[0][0]
    stim_end_sec = stim_sections_sec[-1][1]
    raw = mne.io.read_raw_edf(control_path % subj).pick_channels(channels)
    raw.resample(1000)
    # for cleaning
    # raw.plot_psd()

    # get the 5 minutes before first stim spikes rate (or until the first stim if shorter)
    if stim_start_sec - 60 * 5 < 0:
        baseline_raw = raw.copy().crop(tmin=0, tmax=stim_start_sec)
    else:
        baseline_raw = raw.copy().crop(tmin=stim_start_sec - 60 * 5, tmax=stim_start_sec)
    rates = fill_row(baseline_raw, rates)

    # fill sections of stim and the stops between
    for i, (start, end) in enumerate(stim_sections_sec):
            # fill the current stim
            curr = raw.copy().crop(tmin=start, tmax=end)
            # rates = fill_row(raw.copy().crop(tmin=start, tmax=end), rates, is_stim=True)
            rates = fill_row(curr, rates, is_stim=True)
            if i + 1 < len(stim_sections_sec):
                # the stop is the time between the end of the curr section and the start of the next, buffer of 0.5 sec of the stim
                rates = fill_row(raw.copy().crop(tmin=end + 0.5, tmax=stim_sections_sec[i + 1][0] - 0.5), rates)
            else:
                # 5 min after the last stim
                rates = fill_row(raw.copy().crop(tmin=end + 0.5, tmax=end + 60 * 5), rates)

    results_df = pd.DataFrame(rates)
    file_name = channels[0].replace('SEEG ', '').replace('-REF1' if subj == '396' else '-REF', '')
    results_df.to_csv(f'results/control/{subj}_{file_name}_rates.csv')

    return rates

all_subjects = ['394', '396', '398', '400', '402', '404', '405', '406', '414', '415', '416', '417', '422', '423', '426', '429']
subjects_nrem = {'396': 120, '398': 95, '402': 55, '406': 150, '415': 105, '416': 240, '404': 140}
# for debug
# subj = '416'
# chans = ['RA2', 'RA3']
# detect_subj_chan(subj, chans + [f'SEEG {x}-REF' for x in chans])

for subj in ['404']:
    print(f'subj: {subj}')
    subj_raw = mne.io.read_raw_edf(control_path % subj)
    all_channels = get_control_clean_channels(subj, subj_raw)
    chans_bi = []

    # get the channels for bipolar reference
    for i, chan in enumerate(all_channels):
        if i + 1 < len(all_channels):
            curr = chan.replace('SEEG ', '').replace('-REF1' if subj == '396' else '-REF', '')
            next_chan = all_channels[i + 1]
            next_chan_name = next_chan.replace('SEEG ', '').replace('-REF1' if subj == '396' else '-REF', '')
            # check that its the same contact
            if next_chan_name[:-1] == curr[:-1]:
                chans_bi.append([chan, next_chan])

    # run on each channel and detect the spikes between stims
    for chans in chans_bi:
        rates = detect_subj_chan(subj, chans)
        # plot_subj_chan(rates, chans)
    print(1)
