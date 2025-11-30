import os
import numpy as np
import pandas as pd
import parselmouth
import librosa


def _pitch_stats(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return dict(min=None, max=None, mean=None)
    return dict(min=float(np.min(arr)), max=float(np.max(arr)), mean=float(np.mean(arr)))


def extract_praat_pitch(sound: parselmouth.Sound):
    pitch = sound.to_pitch()
    times = pitch.xs()
    values = pitch.selected_array["frequency"]
    values = np.where(values == 0, np.nan, values)
    stats = _pitch_stats(values)
    return times.tolist(), values.tolist(), stats


def extract_librosa_pitch(file_path: str, hop_length: int = 256):
    y, sr = librosa.load(file_path, sr=None)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),  # ~65 Hz
        fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
        sr=sr,
        hop_length=hop_length,
    )
    f0 = np.array(f0, dtype=float)
    f0[~voiced_flag] = np.nan
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    stats = _pitch_stats(f0)
    return times.tolist(), f0.tolist(), stats


def analyze_wav(file_path):
    snd = parselmouth.Sound(file_path)
    duration = snd.get_total_duration()

    formant = snd.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=5500,
    )

    mid = duration / 2
    F1 = formant.get_value_at_time(1, mid)
    F2 = formant.get_value_at_time(2, mid)

    praat_times, praat_f0, praat_stats = extract_praat_pitch(snd)
    librosa_times, librosa_f0, librosa_stats = extract_librosa_pitch(file_path)

    return {
        "Duration": duration,
        "F1": F1,
        "F2": F2,
        "Praat_Times": praat_times,
        "Praat_F0": praat_f0,
        "Praat_F0_min": praat_stats["min"],
        "Praat_F0_max": praat_stats["max"],
        "Praat_F0_mean": praat_stats["mean"],
        "Librosa_Times": librosa_times,
        "Librosa_F0": librosa_f0,
        "Librosa_F0_min": librosa_stats["min"],
        "Librosa_F0_max": librosa_stats["max"],
        "Librosa_F0_mean": librosa_stats["mean"],
    }


def analyze_all_vowels(base_path, folders):
    rows = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)

        for fname in os.listdir(folder_path):
            if fname.endswith(".wav"):
                fpath = os.path.join(folder_path, fname)

                try:
                    metrics = analyze_wav(fpath)
                    rows.append(
                        {
                            "File": fname,
                            "Vowel": folder,
                            "Path": fpath,
                            **metrics,
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "File": fname,
                            "Vowel": folder,
                            "Duration": None,
                            "F1": None,
                            "F2": None,
                            "Path": f"ERROR: {e}",
                            "Praat_Times": None,
                            "Praat_F0": None,
                            "Praat_F0_min": None,
                            "Praat_F0_max": None,
                            "Praat_F0_mean": None,
                            "Librosa_Times": None,
                            "Librosa_F0": None,
                            "Librosa_F0_min": None,
                            "Librosa_F0_max": None,
                            "Librosa_F0_mean": None,
                        }
                    )

    df = pd.DataFrame(rows)
    return df
