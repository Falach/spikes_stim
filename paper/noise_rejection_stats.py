# %% [markdown]
# # Noise Rejection Statistics
#
# Per-subject and group-level summary of:
# - **Excluded channels**: count and % of total channels
# - **Bad epochs**: rejected time in seconds, minutes, and % of experiment duration
#
# Experiment duration is defined as: sleep baseline onset → end of last pause block.
# Computed separately for the **stimulation group** and the **control group**.

# %%
import numpy as np
import pandas as pd
import os, sys
import mne

sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))
import utils

# ============================================================
# Subject lists and paths
# ============================================================
stim_subjects = ['485', '486', '487', '488', '497', '498', '499',
                 '505', '5107', '515', '520', '545']

control_subjects = ['394', '396', '398', '402', '404', '405',
                    '406', '415', '416', '417']

base_path = r"D:\Ofer_backup\V6_model"
bad_channels_dir = r"D:\Ofer_backup\bad_channels"

# Stim feature path template
stim_feat_template = os.path.join(
    base_path, "p{subj}", "unichannel_model", "min_amplitude_1",
    "features", "p{subj}_flat_features.csv"
)

# Control feature path template
ctrl_feat_template = os.path.join(
    base_path, "controls", "p{subj}", "unichannel_model", "min_amplitude_1",
    "features", "p{subj}_flat_features.csv"
)

# Stim annotations template
stim_ann_template = r"D:\Maya\clean Stim Project\P{subj}\P{subj}_annotations_ms.csv"

# Control annotations template
ctrl_ann_template = r"D:\UCLA\{subj}_clean_mtl_annot.fif"
ctrl_ann_405 = r"D:\UCLA\405_clean_mtl_annot_cropped.fif"

# %%
# ============================================================
# Improved function: compute rejection statistics
# ============================================================
def compute_rejection_stats(
    subject_id,
    df_features,
    annotations_path,
    experiment_duration_sec,
    ts_col="timestamp",
    channel_col="channel",
    bad_channels_dir_local=bad_channels_dir,
):
    """
    Analyse the annotation and bad-channel files for a subject and
    return a dict of rejection statistics.

    Improvements over earlier prototype:
    - Reports rejected time in both seconds AND minutes
    - Reports experiment duration in both seconds AND minutes
    - Handles both CSV and FIF annotation formats
    - Merges overlapping annotation intervals correctly
    - Only counts global annotations for rejected-time (channel-specific
      annotations do not block the entire time window)
    - Gracefully handles missing files with warnings
    """
    df = df_features.copy()
    all_channels = df[channel_col].astype(str).unique()

    bad_channel_names = set()
    rejected_seconds_annot = 0.0

    # ── Annotations ───────────────────────────────────────────
    # Simply sum the duration column across ALL annotations.
    if annotations_path is not None and os.path.exists(annotations_path):
        if annotations_path.lower().endswith('.fif'):
            raw_info = mne.io.read_raw_fif(annotations_path, preload=False)
            # duration is already in seconds
            rejected_seconds_annot = float(np.sum(raw_info.annotations.duration))
        else:
            ann = pd.read_csv(annotations_path)
            # duration column is in seconds
            rejected_seconds_annot = float(ann["duration"].astype(float).sum())
    elif annotations_path is not None:
        print(f"  Warning: annotations file not found: {annotations_path}")

    # ── Bad channels ──────────────────────────────────────────
    if bad_channels_dir_local is not None and subject_id is not None:
        bad_channels_path = os.path.join(
            bad_channels_dir_local, f"p{subject_id}_bad_channels.txt"
        )
        if os.path.exists(bad_channels_path):
            with open(bad_channels_path, "r", encoding="utf-8") as f:
                bad_channel_names = {
                    line.strip() for line in f if line.strip()
                }
        else:
            print(f"  Warning: bad-channels file not found: {bad_channels_path}")

    # ── Compute statistics ────────────────────────────────────
    # Report ALL bad channels from the txt file, not just
    # those that appear in the feature data.  Some bad channels
    # may have been stripped by the upstream detection pipeline
    # and therefore don't show up in df, but they are still
    # channels that were excluded from analysis.
    channels_in_data = set(all_channels)
    n_bad_channels = len(bad_channel_names)

    # True total = channels present in data + any bad channels
    # that are NOT already in the data (i.e. they were removed
    # before the features file was written).
    missing_bad = bad_channel_names - channels_in_data
    n_total_channels_true = len(channels_in_data) + len(missing_bad)

    pct_bad_channels = (100.0 * n_bad_channels / n_total_channels_true
                        if n_total_channels_true > 0 else 0.0)

    rejected_minutes = rejected_seconds_annot / 60.0
    pct_time_rejected = (100.0 * rejected_seconds_annot / experiment_duration_sec
                         if experiment_duration_sec > 0 else 0.0)
    experiment_duration_min = experiment_duration_sec / 60.0

    return {
        "subject_id":             subject_id,
        "experiment_duration_sec": round(experiment_duration_sec, 2),
        "experiment_duration_min": round(experiment_duration_min, 2),
        "n_total_channels":       n_total_channels_true,
        "n_channels_in_data":     len(channels_in_data),
        "n_bad_channels":         n_bad_channels,
        "bad_channels_pct":       round(pct_bad_channels, 2),
        "rejected_seconds":       round(rejected_seconds_annot, 2),
        "rejected_minutes":       round(rejected_minutes, 2),
        "rejected_time_pct":      round(pct_time_rejected, 2),
        "bad_channel_names":      sorted(bad_channel_names),
    }


# %%
# ============================================================
# Helper: compute experiment duration (baseline → last pause)
# ============================================================
def get_experiment_duration_stim(subj):
    """
    For stimulation subjects.
    Duration = min(nrem_start, stim_start) → stim_end + 5 min
    Handles subjects where baseline is negative (515, 545).
    """
    nrem_epochs, _, stim_epochs = utils.get_nrem_epochs(subj)
    nrem_start  = nrem_epochs[0][0]
    stim_start  = stim_epochs[0][0]
    stim_end    = stim_epochs[-1][1]
    experiment_start = min(nrem_start, stim_start)
    experiment_end   = stim_end + 5 * 60          # last pause = 5 min
    return experiment_end - experiment_start


def get_experiment_duration_control(subj):
    """
    For control subjects.
    Duration = min(nrem_start, protocol_start) → protocol_end + 5 min
    """
    nrem_epochs  = utils.get_control_nrem_epochs(subj)
    stim_epochs  = utils.get_stim_starts_upgraded(subj)
    if len(stim_epochs) == 0 or len(nrem_epochs) == 0:
        print(f"  Warning: no data for control {subj}")
        return None
    nrem_start  = nrem_epochs[0][0]
    stim_start  = stim_epochs[0][0]
    stim_end    = stim_epochs[-1][1]
    experiment_start = min(nrem_start, stim_start)
    experiment_end   = stim_end + 5 * 60
    return experiment_end - experiment_start


# %%
# ============================================================
# Main processing loop
# ============================================================
all_stats = []

# ── Stimulation group ─────────────────────────────────────────
print("=== Processing STIMULATION group ===")
for subj in stim_subjects:
    feat_path = stim_feat_template.format(subj=subj)
    if not os.path.exists(feat_path):
        print(f"  Missing feature file for stim subject {subj}, skipping")
        continue

    df = pd.read_csv(feat_path)
    # Keep only columns needed for channel counting & timestamp matching
    df = df.drop_duplicates(subset="group", keep="first")

    ann_path = stim_ann_template.format(subj=subj)

    # Compute experiment duration
    exp_dur = get_experiment_duration_stim(subj)

    print(f"  Subject {subj}: experiment duration = {exp_dur/60:.1f} min")

    stats = compute_rejection_stats(
        subj, df, ann_path,
        experiment_duration_sec=exp_dur,
        ts_col="timestamp", channel_col="channel",
    )
    stats["group"] = "stim"
    all_stats.append(stats)

print()

# ── Control group ─────────────────────────────────────────────
print("=== Processing CONTROL group ===")
for subj in control_subjects:
    feat_path = ctrl_feat_template.format(subj=subj)
    if not os.path.exists(feat_path):
        print(f"  Missing feature file for control subject {subj}, skipping")
        continue

    df = pd.read_csv(feat_path)
    df = df.drop_duplicates(subset="group", keep="first")

    # Control annotation path (special case for 405)
    if subj == '405':
        ann_path = ctrl_ann_405
    else:
        ann_path = ctrl_ann_template.format(subj=subj)

    # Compute experiment duration
    exp_dur = get_experiment_duration_control(subj)
    if exp_dur is None:
        print(f"  Skipping control subject {subj} (no epochs)")
        continue

    print(f"  Subject {subj}: experiment duration = {exp_dur/60:.1f} min")

    stats = compute_rejection_stats(
        subj, df, ann_path,
        experiment_duration_sec=exp_dur,
        ts_col="timestamp", channel_col="channel",
    )
    stats["group"] = "control"
    all_stats.append(stats)


# %%
# ============================================================
# Per-patient summary table
# ============================================================
display_cols = [
    "group", "subject_id",
    "experiment_duration_min",
    "n_total_channels", "n_channels_in_data", "n_bad_channels", "bad_channels_pct",
    "rejected_seconds", "rejected_minutes", "rejected_time_pct",
    "bad_channel_names",
]

summary_df = pd.DataFrame(all_stats)[display_cols]
summary_df = summary_df.sort_values(["group", "subject_id"]).reset_index(drop=True)

print("=" * 80)
print("  PER-PATIENT REJECTION SUMMARY")
print("=" * 80)
print(summary_df.to_string(index=False))
print()

# %%
# ============================================================
# Group-level statistics  (mean ± SD)
# ============================================================
numeric_cols = [
    "experiment_duration_min",
    "n_total_channels", "n_channels_in_data", "n_bad_channels", "bad_channels_pct",
    "rejected_seconds", "rejected_minutes", "rejected_time_pct",
]

def group_summary(df_group, label):
    """Return a summary DataFrame with mean and std for a group."""
    agg = df_group[numeric_cols].agg(["mean", "std", "min", "max"]).T
    agg.columns = ["mean", "std", "min", "max"]
    agg = agg.round(2)
    print(f"\n{'=' * 60}")
    print(f"  {label}  (n = {len(df_group)})")
    print(f"{'=' * 60}")
    for col in numeric_cols:
        m = agg.loc[col, "mean"]
        s = agg.loc[col, "std"]
        print(f"  {col:30s}  {m:10.2f} ± {s:.2f}")
    return agg


df_stim = summary_df[summary_df["group"] == "stim"]
df_ctrl = summary_df[summary_df["group"] == "control"]

summary_stim = group_summary(df_stim, "STIMULATION GROUP")
summary_ctrl = group_summary(df_ctrl, "CONTROL GROUP")
summary_all  = group_summary(summary_df, "BOTH GROUPS COMBINED")

# %%
# ============================================================
# Display final summary DataFrames
# ============================================================
print("\n\nStimulation group statistics:")
print(summary_stim)

print("\n\nControl group statistics:")
print(summary_ctrl)

print("\n\nCombined statistics:")
print(summary_all)
