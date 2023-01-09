import mne
import pandas as pd
import numpy as np
import h5py
import glob
import os
from datetime import datetime, timezone, timedelta
import pyedflib
from pyedflib._extensions._pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS


def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    if utc_stamp is None:
        return datetime.now()
    if 'datetime' in str(type(utc_stamp)): return utc_stamp
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def rotem_write_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, overwrite=True):
    conversion_time = datetime.now()
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')

    # static settings
    has_annotations = True if len(mne_raw.annotations) > 0 else False
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = FILETYPE_EDFPLUS if has_annotations else FILETYPE_EDF
        dmin, dmax = -32768, 32767
    else:
        file_type = FILETYPE_BDFPLUS if has_annotations else FILETYPE_BDF
        dmin, dmax = -8388608, 8388607

    print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = mne_raw.info['sfreq']
    date = _stamp_to_dt(mne_raw.info['meas_date'])

    if tmin:
        date += timedelta(seconds=tmin)
    first_sample = int(sfreq * tmin)
    last_sample = int(sfreq * tmax) if tmax is not None else None

    # convert data
    channels = mne_raw.get_data(picks,
                                start=first_sample,
                                stop=last_sample)

    # convert to microvolts to scale up precision
    # channels *= 1e6

    # set conversion parameters
    n_channels = len(channels)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels,
                               file_type=file_type)

        channel_info = []

        ch_idx = range(n_channels) if picks is None else picks
        for i in ch_idx:
            ch_dict = {'label': mne_raw.ch_names[i],
                       'dimension': 'uV',
                       'sample_rate': sfreq,
                       'physical_min': -5000,
                       'physical_max': 5000,
                       # 'physical_min': np.nanmin(channels[i]),
                       # 'physical_max': np.nanmax(channels[i]) + 1,
                       'digital_min': dmin,
                       'digital_max': dmax,
                       'transducer': '',
                       'prefilter': ''}

            channel_info.append(ch_dict)

        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(channels)
        for annotation in mne_raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            description = annotation['description']
            f.writeAnnotation(onset, duration, description)

    except Exception as e:
        raise e
    finally:
        f.close()
        print('conversion time:')
        print(datetime.now() - conversion_time)
    return True


subj = '488'
save_night = True
save_tag = True
subj_stim = {'485': 9595, '486': 4191, '487': 6314, '488': 9389, '489': 8743, '490': 7528, '496': 3252, '497': 5530,
             '498': 4583, '499': 7965, '505': 11320, '510-1': 9736, '510-7': 5629, '515': 490, '520': 1400, '538': 5100,
             '541': 5160, '544': 2535, '545': 8400}
first_stim_sec = subj_stim[subj]
mne_raw = None
subj_files_list = glob.glob(f'D:\\Maya\\p{subj}\\MACRO\\*')
for curr_file in subj_files_list:
    try:
        f = h5py.File(curr_file, 'r')
        data = f.get('data')
        data = np.array(data)
        # replace nan with previous value
        df = pd.DataFrame(data)
        df = df.fillna(method="ffill")
        data = df.to_numpy().T
        # ch_names = [x.replace('\'', '') for x in ch_names.iloc[:, 0].tolist()]
        # ch_names = [x for x in range(10)]
        ch_name_array = np.array([x for x in f['LocalHeader/origName']], dtype='uint16')
        ch_name = ''
        for x in ch_name_array:
            ch_name += chr(x[0])
        # don't include scalp channels
        if ch_name[0] in ['R', 'L']:
            # sfreq = int(np.array(f.get('hdr/Fs'))[0][0])
            sfreq = np.array(f['LocalHeader/samplingRate'])[0][0]
            info = mne.create_info(ch_names=[ch_name], sfreq=sfreq)
            if mne_raw is None:
                mne_raw = mne.io.RawArray(data, info)
            else:
                mne_raw.add_channels([mne.io.RawArray(data, info)])
    except Exception as e:
        pass

# mne_raw.load_data()
mne_raw.reorder_channels(sorted(mne_raw.ch_names))
# save night
if save_night:
    annotation = mne.Annotations(onset=[first_stim_sec],
                                  duration=[0],
                                  description=['stim_start'])
    mne_raw.set_annotations(annotation)
    mne_raw.save(f'D:\\Maya\\p{subj}\\P{subj}_fixed.fif', overwrite=True)
    rotem_write_edf(mne_raw, f'D:\\Maya\\p{subj}\\P{subj}_fixed.edf')

# save for tag
if save_tag:
    for_tag = mne_raw.copy().crop(tmin=first_stim_sec - 180, tmax=first_stim_sec + 180)
# before_stim = mne_raw.copy().crop(tmin=first_stim_sec - (18 * 60), tmax=first_stim_sec - (15 * 60))
# during_stim = mne_raw.copy().crop(tmin=first_stim_sec - 5, tmax=first_stim_sec + 180)
# mne_raw.plot()
    rotem_write_edf(for_tag, f'D:\\Maya\\p{subj}\\P{subj}_for_tag.edf')
# rotem_write_edf(during_stim, f'C:\\Maya\\p{subj}\\P{subj}_during_stim.edf')

print()
