import os

# macOS stability
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import parselmouth


def estimate_formants(sound, time):
    """
    Returns F1, F2, F3 at a given time using Burg algorithm (Praat).
    """
    formant = sound.to_formant_burg()
    f1 = formant.get_value_at_time(1, time)
    f2 = formant.get_value_at_time(2, time)
    f3 = formant.get_value_at_time(3, time)
    return f1, f2, f3


def generate_vowel_space_plot(df, save_path):
    plt.figure(figsize=(7, 6))

    for vowel, group in df.groupby("Vowel"):
        plt.scatter(group["F2"], group["F1"], label=vowel, alpha=0.8)
        plt.scatter(group["F2"].mean(), group["F1"].mean(), marker="x", s=120)

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel("F2 (Hz)")
    plt.ylabel("F1 (Hz)")
    plt.title("Vowel Space Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_vowel_boxplots(df, save_path):
    plt.figure(figsize=(8, 5))
    vowels = sorted(df["Vowel"].unique())
    positions = np.arange(len(vowels))

    f1_data = [df[df["Vowel"] == v]["F1"].dropna() for v in vowels]
    f2_data = [df[df["Vowel"] == v]["F2"].dropna() for v in vowels]

    plt.boxplot(
        f1_data,
        positions=positions - 0.15,
        widths=0.25,
        patch_artist=True,
        boxprops=dict(facecolor="#c7d2fe"),
        medianprops=dict(color="#1e3a8a"),
        labels=[""] * len(vowels),
    )

    plt.boxplot(
        f2_data,
        positions=positions + 0.15,
        widths=0.25,
        patch_artist=True,
        boxprops=dict(facecolor="#a5f3fc"),
        medianprops=dict(color="#0f172a"),
        labels=vowels,
    )

    plt.xticks(positions, vowels)
    plt.ylabel("Frequency (Hz)")
    plt.title("F1 / F2 Distribution by Vowel")
    plt.legend(["F1", "F2"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_duration_histogram(df, save_path):
    plt.figure(figsize=(7, 4))
    durations = df["Duration"].dropna()
    plt.hist(durations, bins=15, color="#93c5fd", edgecolor="#1d4ed8", alpha=0.9)
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Recording Duration Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_spectrogram(file_path, save_path):
    # Load audio via librosa
    y, sr = librosa.load(file_path, sr=None)
    duration = len(y) / sr

    # Load with parselmouth for formants & pitch
    snd = parselmouth.Sound(file_path)

    # Setup figure: Waveform + Spectrogram
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 3]})

    # --------------------------------------------------
    # 1) Waveform
    # --------------------------------------------------
    librosa.display.waveshow(y, sr=sr, ax=ax[0], color="#00e6ac")
    ax[0].set_title("Waveform", fontsize=13)
    ax[0].grid(color="gray", linestyle="--", linewidth=0.3, alpha=0.3)

    # --------------------------------------------------
    # 2) Mel Spectrogram
    # --------------------------------------------------
    n_fft = 2048
    hop = 256

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=128,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax[1]
    )
    fig.colorbar(img, ax=ax[1], format="%+2.f dB")
    ax[1].set_title("Mel Spectrogram + Pitch + Formants", fontsize=13)

    # --------------------------------------------------
    # 3) Pitch (F0 contour)
    # --------------------------------------------------
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values[pitch_values == 0] = np.nan
    times_pitch = pitch.xs()

    ax[1].plot(times_pitch, pitch_values, color="#00ff99", linewidth=2, label="Pitch (F0)")

    # --------------------------------------------------
    # 4) Formants (F1, F2, F3)
    # --------------------------------------------------
    times = np.linspace(0, duration, len(pitch_values))
    f1_list, f2_list, f3_list = [], [], []

    formant = snd.to_formant_burg()

    for t in times:
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        f1_list.append(f1 if f1 else np.nan)
        f2_list.append(f2 if f2 else np.nan)
        f3_list.append(f3 if f3 else np.nan)

    ax[1].plot(times, f1_list, color="#1e90ff", linewidth=1.4, label="F1")
    ax[1].plot(times, f2_list, color="#ffcc00", linewidth=1.4, label="F2")
    ax[1].plot(times, f3_list, color="#ff66cc", linewidth=1.4, label="F3")

    ax[1].legend(loc="upper right")

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=240)
    plt.close()


def generate_pitch_comparison_plot(
    praat_times,
    praat_f0,
    librosa_times,
    librosa_f0,
    save_path,
    title=None,
):
    plt.figure(figsize=(10, 4))
    if praat_times and praat_f0:
        plt.plot(praat_times, praat_f0, label="Praat (parselmouth)", color="#2563eb", linewidth=1.8)
    if librosa_times and librosa_f0:
        plt.plot(librosa_times, librosa_f0, label="Librosa (pyin)", color="#ea580c", linewidth=1.4, alpha=0.9)

    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title(title or "Pitch Contour Comparison")
    plt.legend()
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
