
import mne
import pandas as pd
import numpy as np
from visbrain.gui import Sleep

subj = '416'
edf = f'C:\\Lilach\\{subj}_for_tag_filtered_fix_tag.edf'
raw = mne.io.read_raw_edf(edf).resample(1000)
sf = raw.info['sfreq']
real_df = pd.read_csv(f"C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\real_AH_{subj}.csv")
rf_df = pd.read_csv(f"C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\rf_AH_loo_{subj}.csv")
lgbm_df = pd.read_csv(f"C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\lgbm_AH_loo_{subj}.csv")
maya_df = pd.read_csv(f"C:\\Users\\user\\PycharmProjects\\pythonProject\\results\\maya_AH_bi_{subj}.csv")

real = np.zeros((1, raw.n_times))
for evt in real_df.values:
        real[0][250 * evt[0]: 250 * evt[0] + 250] = 1

rf = np.zeros((1, raw.n_times))
for evt in rf_df.values:
        rf[0][250 * evt[0]: 250 * evt[0] + 250] = 1

lgbm = np.zeros((1, raw.n_times))
for evt in lgbm_df.values:
        lgbm[0][250 * evt[0]: 250 * evt[0] + 250] = 1

maya = np.zeros((1, raw.n_times))
for evt in maya_df.values:
        maya[0][250 * evt[0]: 250 * evt[0] + 250] = 1

raw.pick_channels(['LAH1', 'LAH1-LAH2', 'RAH1',  'RAH1-RAH2'])
info_real = mne.create_info(['real'], raw.info['sfreq'], ['stim'])
info_rf = mne.create_info(['rf'], raw.info['sfreq'], ['stim'])
info_lgbm = mne.create_info(['lgbm'], raw.info['sfreq'], ['stim'])
info_maya = mne.create_info(['thresh'], raw.info['sfreq'], ['stim'])
real_raw = mne.io.RawArray(real, info_real)
rf_raw = mne.io.RawArray(rf, info_rf)
lgbm_raw = mne.io.RawArray(lgbm, info_lgbm)
maya_raw = mne.io.RawArray(maya, info_maya)
raw.load_data()
raw.add_channels([real_raw, rf_raw, lgbm_raw, maya_raw], force_update_info=True)

sp = Sleep(data=raw._data, sf=raw.info['sfreq'], channels=raw.info['ch_names'], downsample=None)
# sp.replace_detections('peak', peak_index)
# sp.replace_detections('spindle', spikes_index)
sp.show()
print('finish')


# edf = 'C:\\Lilach\\402_for_tag.edf'
#
# raw = mne.io.read_raw_edf(edf)
#
# data, sf, chan = raw.get_data(), raw.info['sfreq'], raw.info['ch_names']
#
# Sleep(data=data, sf=sf, channels=chan, annotations=raw.annotations).show()
#
# print(1)