"""
Vowel synthesis toolkit.

Implements a simple source-filter pipeline:
1) Glottal source (impulse train by default).
2) Cascaded 2nd-order IIR resonators for formants.
3) Saving / plotting helpers plus formant estimation via parselmouth.

Defaults:
    fs = 16000
    duration = 1.0 second
Dependencies: numpy, scipy, matplotlib, soundfile, librosa, parselmouth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import soundfile as sf
from scipy.signal import butter, filtfilt, iirpeak, lfilter


FS = 16000
DURATION = 1.0


@dataclass
class VowelSpec:
    name: str
    formants: Sequence[Tuple[float, float]]
    f0: float = 120.0
    jitter: float = 0.0


VOWEL_PRESETS: Dict[str, VowelSpec] = {
    "i": VowelSpec("i", [(300, 70), (2400, 120), (3000, 200)]),
    "e": VowelSpec("e", [(500, 70), (1900, 120), (2500, 200)]),
    "a": VowelSpec("a", [(800, 70), (1200, 120), (2600, 200)]),
    "o": VowelSpec("o", [(500, 70), (900, 120), (2400, 200)]),
    "u": VowelSpec("u", [(350, 70), (700, 120), (2400, 200)]),
}


# ──────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────
def impulse_train(f0: float, fs: int = FS, duration: float = DURATION, lp_hz: float | None = 400.0) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = np.zeros_like(t)
    period = max(1, int(fs / f0))
    sig[::period] = 1.0
    if lp_hz:
        b, a = butter(2, lp_hz / (fs / 2))
        sig = filtfilt(b, a, sig)
    return sig


# Placeholder for a future LF model; left simple to keep the core path lightweight.
def lf_glottal_placeholder(*args, **kwargs) -> np.ndarray:
    return impulse_train(kwargs.get("f0", 120.0), kwargs.get("fs", FS), kwargs.get("duration", DURATION))


# ──────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────
def resonator(center_freq: float, bandwidth: float, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    Q = center_freq / bandwidth
    return iirpeak(center_freq / (fs / 2), Q)


def apply_resonators(x: np.ndarray, formants: Sequence[Tuple[float, float]], fs: int) -> np.ndarray:
    y = np.array(x, dtype=float)
    for f, bw in formants:
        b, a = resonator(f, bw, fs)
        y = lfilter(b, a, y)
    return y


# ──────────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────────
def estimate_formants(path: str) -> Tuple[float, float]:
    snd = parselmouth.Sound(path)
    formant = snd.to_formant_burg()
    mid = snd.get_total_duration() / 2
    f1 = formant.get_value_at_time(1, mid)
    f2 = formant.get_value_at_time(2, mid)
    return f1, f2


def synthesize_vowel(
    vowel: VowelSpec,
    fs: int = FS,
    duration: float = DURATION,
    source_kind: str = "impulse",
) -> np.ndarray:
    if source_kind == "impulse":
        src = impulse_train(vowel.f0, fs=fs, duration=duration)
    elif source_kind == "lf":
        src = lf_glottal_placeholder(f0=vowel.f0, fs=fs, duration=duration)
    else:
        raise ValueError(f"Unknown source {source_kind}")
    y = apply_resonators(src, vowel.formants, fs)
    y = y / np.max(np.abs(y)) * 0.9
    return y


def save_and_plot(
    signal: np.ndarray,
    fs: int,
    out_wav: str,
    out_fig: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    sf.write(out_wav, signal, fs)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 1.6]})
    t = np.arange(len(signal)) / fs
    axs[0].plot(t, signal, color="#2563eb")
    axs[0].set_title(f"Waveform — {title}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(alpha=0.3, linestyle="--")

    S = librosa.stft(signal, n_fft=1024, hop_length=256, window="hann")
    S_db = librosa.power_to_db(np.abs(S) ** 2, ref=np.max)
    img = librosa.display.specshow(
        S_db,
        sr=fs,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=axs[1],
    )
    fig.colorbar(img, ax=axs[1], format="%+2.f dB")
    axs[1].set_title("Spectrogram")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)


def batch_synthesize(output_dir: str = "synth_outputs", vowels: Iterable[str] | None = None, fs: int = FS, duration: float = DURATION) -> List[dict]:
    vowels = vowels or list(VOWEL_PRESETS.keys())
    rows = []
    for name in vowels:
        spec = VOWEL_PRESETS[name]
        y = synthesize_vowel(spec, fs=fs, duration=duration)
        wav_path = os.path.join(output_dir, f"synth_{name}.wav")
        fig_path = os.path.join(output_dir, f"synth_{name}.png")
        save_and_plot(y, fs, wav_path, fig_path, title=f"/{name}/ (F0={spec.f0} Hz)")
        f1, f2 = estimate_formants(wav_path)
        rows.append({"Vowel": name, "Wav": wav_path, "Plot": fig_path, "F1": f1, "F2": f2})
    return rows


def vowel_space_comparison(natural_paths: Dict[str, str], synth_rows: List[dict], out_path: str = "synth_outputs/vowel_space_compare.png"):
    import matplotlib.pyplot as plt

    syn_df = []
    for r in synth_rows:
        syn_df.append((r["Vowel"], r["F1"], r["F2"], "Synthetic"))

    nat_df = []
    for vowel, path in natural_paths.items():
        f1, f2 = estimate_formants(path)
        nat_df.append((vowel, f1, f2, "Natural"))

    plt.figure(figsize=(7, 6))
    for label, df in (("Synthetic", syn_df), ("Natural", nat_df)):
        xs = [d[2] for d in df]
        ys = [d[1] for d in df]
        plt.scatter(xs, ys, label=label, alpha=0.8)
        for (v, f1, f2, _) in df:
            plt.text(f2 + 20, f1 + 20, v, fontsize=9)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel("F2 (Hz)")
    plt.ylabel("F1 (Hz)")
    plt.title("Vowel Space: Synthetic vs Natural")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def detailed_vowel_comparison(data_path: str, synth_rows: List[dict], output_dir: str = "synth_outputs/comparisons"):
    """
    Compare each synthetic vowel with 2 natural samples.
    Creates side-by-side spectrogram and formant comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Map synthetic vowel names to data folder names
    vowel_folder_map = {
        "i": "heed",
        "e": "hayed", 
        "a": "hod",
        "o": "hoed",
        "u": "who_d"
    }
    
    comparison_results = []
    
    for synth in synth_rows:
        vowel = synth["Vowel"]
        folder_name = vowel_folder_map.get(vowel)
        
        if not folder_name:
            continue
            
        folder_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Get first 2 natural samples
        natural_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".wav")])[:2]
        
        if len(natural_files) < 2:
            continue
        
        natural_paths = [os.path.join(folder_path, f) for f in natural_files]
        
        # Extract formants for comparison
        synth_f1, synth_f2 = synth["F1"], synth["F2"]
        
        natural_formants = []
        for nat_path in natural_paths:
            f1, f2 = estimate_formants(nat_path)
            natural_formants.append({"path": nat_path, "F1": f1, "F2": f2, "filename": os.path.basename(nat_path)})
        
        # Calculate formant errors
        avg_nat_f1 = np.mean([n["F1"] for n in natural_formants])
        avg_nat_f2 = np.mean([n["F2"] for n in natural_formants])
        f1_error = abs(synth_f1 - avg_nat_f1)
        f2_error = abs(synth_f2 - avg_nat_f2)
        
        # Create comparison plot
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Synthetic spectrogram (top row, span 3 columns)
        ax_synth = fig.add_subplot(gs[0, :])
        synth_signal, synth_sr = librosa.load(synth["Wav"], sr=None)
        S_synth = librosa.stft(synth_signal, n_fft=1024, hop_length=256, window="hann")
        S_synth_db = librosa.power_to_db(np.abs(S_synth) ** 2, ref=np.max)
        librosa.display.specshow(S_synth_db, sr=synth_sr, hop_length=256, x_axis="time", y_axis="hz", cmap="magma", ax=ax_synth)
        ax_synth.set_title(f"Synthetic /{vowel}/ — F1={synth_f1:.1f} Hz, F2={synth_f2:.1f} Hz", fontsize=12, fontweight='bold')
        
        # Natural sample 1 (middle row, left)
        ax_nat1 = fig.add_subplot(gs[1, 0:2])
        nat1_signal, nat1_sr = librosa.load(natural_formants[0]["path"], sr=None)
        S_nat1 = librosa.stft(nat1_signal, n_fft=1024, hop_length=256, window="hann")
        S_nat1_db = librosa.power_to_db(np.abs(S_nat1) ** 2, ref=np.max)
        librosa.display.specshow(S_nat1_db, sr=nat1_sr, hop_length=256, x_axis="time", y_axis="hz", cmap="viridis", ax=ax_nat1)
        ax_nat1.set_title(f"Natural 1: {natural_formants[0]['filename']} — F1={natural_formants[0]['F1']:.1f} Hz, F2={natural_formants[0]['F2']:.1f} Hz", fontsize=10)
        
        # Natural sample 2 (bottom row, left)
        ax_nat2 = fig.add_subplot(gs[2, 0:2])
        nat2_signal, nat2_sr = librosa.load(natural_formants[1]["path"], sr=None)
        S_nat2 = librosa.stft(nat2_signal, n_fft=1024, hop_length=256, window="hann")
        S_nat2_db = librosa.power_to_db(np.abs(S_nat2) ** 2, ref=np.max)
        librosa.display.specshow(S_nat2_db, sr=nat2_sr, hop_length=256, x_axis="time", y_axis="hz", cmap="viridis", ax=ax_nat2)
        ax_nat2.set_title(f"Natural 2: {natural_formants[1]['filename']} — F1={natural_formants[1]['F1']:.1f} Hz, F2={natural_formants[1]['F2']:.1f} Hz", fontsize=10)
        
        # Formant comparison plot (right side)
        ax_formant = fig.add_subplot(gs[1:, 2])
        
        # Plot formant values
        formants = ['F1', 'F2']
        x = np.arange(len(formants))
        width = 0.25
        
        synth_vals = [synth_f1, synth_f2]
        nat1_vals = [natural_formants[0]['F1'], natural_formants[0]['F2']]
        nat2_vals = [natural_formants[1]['F1'], natural_formants[1]['F2']]
        
        ax_formant.bar(x - width, synth_vals, width, label='Synthetic', color='#ff6b6b')
        ax_formant.bar(x, nat1_vals, width, label='Natural 1', color='#4ecdc4')
        ax_formant.bar(x + width, nat2_vals, width, label='Natural 2', color='#95e1d3')
        
        ax_formant.set_ylabel('Frequency (Hz)')
        ax_formant.set_title('Formant Comparison', fontweight='bold')
        ax_formant.set_xticks(x)
        ax_formant.set_xticklabels(formants)
        ax_formant.legend()
        ax_formant.grid(axis='y', alpha=0.3)
        
        # Add error text
        error_text = f'Avg F1 Error: {f1_error:.1f} Hz\nAvg F2 Error: {f2_error:.1f} Hz'
        ax_formant.text(0.5, 0.02, error_text, transform=ax_formant.transAxes, 
                       fontsize=9, verticalalignment='bottom', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f"Vowel /{vowel}/ — Synthetic vs Natural Comparison", fontsize=14, fontweight='bold')
        
        # Save comparison plot
        comparison_path = os.path.join(output_dir, f"comparison_{vowel}.png")
        plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        comparison_results.append({
            "Vowel": vowel,
            "Synth_F1": synth_f1,
            "Synth_F2": synth_f2,
            "Natural1_F1": natural_formants[0]["F1"],
            "Natural1_F2": natural_formants[0]["F2"],
            "Natural2_F1": natural_formants[1]["F1"],
            "Natural2_F2": natural_formants[1]["F2"],
            "Avg_Natural_F1": avg_nat_f1,
            "Avg_Natural_F2": avg_nat_f2,
            "F1_Error": f1_error,
            "F2_Error": f2_error,
            "ComparisonPlot": comparison_path,
            "Natural1_File": natural_formants[0]["filename"],
            "Natural2_File": natural_formants[1]["filename"]
        })
    
    # Create summary comparison plot (vowel space with all samples)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for result in comparison_results:
        vowel = result["Vowel"]
        # Plot synthetic
        ax.scatter(result["Synth_F2"], result["Synth_F1"], 
                  marker='o', s=150, label=f'Synth /{vowel}/', alpha=0.8)
        # Plot natural samples
        ax.scatter(result["Natural1_F2"], result["Natural1_F1"], 
                  marker='x', s=100, alpha=0.6)
        ax.scatter(result["Natural2_F2"], result["Natural2_F1"], 
                  marker='x', s=100, alpha=0.6)
        # Add vowel label
        ax.text(result["Synth_F2"] + 30, result["Synth_F1"] + 30, vowel, fontsize=10, fontweight='bold')
    
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2 (Hz)", fontsize=12)
    ax.set_ylabel("F1 (Hz)", fontsize=12)
    ax.set_title("Vowel Space: Synthetic (○) vs Natural Samples (×)", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    
    summary_path = os.path.join(output_dir, "vowel_space_all_samples.png")
    plt.savefig(summary_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    
    return comparison_results, summary_path


if __name__ == "__main__":
    # Example usage: synthesize default vowels and compare with a provided natural set.
    synth_rows = batch_synthesize()
    # Example placeholder for natural comparisons: supply your recorded vowels here.
    # natural = {"a": "Data/a/1.wav", "i": "Data/i/1.wav", ...}
    # vowel_space_comparison(natural, synth_rows)
    print("Synthesis complete. Outputs in synth_outputs/")
