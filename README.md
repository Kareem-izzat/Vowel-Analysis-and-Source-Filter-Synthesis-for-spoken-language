# Spoken Language Processing - Vowel Analysis and Synthesis

A Flask-based web application for acoustic analysis of vowel sounds, featuring formant extraction, pitch analysis, and source-filter synthesis with detailed comparison capabilities.

## Features

### Part A: Vowel Analysis
- Extracts F1, F2, and F3 formants using Praat/Parselmouth
- Calculates vowel duration
- Generates spectrograms with formant overlays
- Analyzes 50 vowel tokens (10 samples × 5 vowels: /i/, /e/, /a/, /o/, /u/)
- Exports results to CSV format

### Part B: Pitch Analysis
- Dual pitch tracking methods:
  - **Praat method**: Using Parselmouth's autocorrelation
  - **Librosa method**: Using pyin() algorithm
- Visualizes pitch contours over time
- Compares both methods side-by-side

### Part C: Vowel Synthesis & Comparison
- Source-filter synthesis using impulse train + cascaded IIR formant filters
- Synthesizes all 5 vowels with extracted formant values
- **Detailed Comparison Feature**:
  - Compares each synthetic vowel with 2 natural samples
  - Side-by-side spectrograms
  - Formant bar charts showing F1/F2 accuracy
  - Formant error analysis (Hz differences)
  - Vowel space visualization (F1 vs F2 plot)

## Project Structure

```
SpokenA1/
├── app.py                  # Flask application with routing
├── synth.py               # Core synthesis and comparison logic
├── requirements.txt       # Python dependencies
├── run_comparison.py      # Standalone script for batch comparisons
├── COMPARISON_README.md   # Detailed comparison feature docs
├── Data/                  # Natural vowel recordings
│   ├── hayed/            # /e/ tokens
│   ├── heed/             # /i/ tokens
│   ├── hod/              # /a/ tokens
│   ├── hoed/             # /o/ tokens
│   └── who_d/            # /u/ tokens
├── static/
│   ├── results.csv       # Vowel analysis results
│   ├── plots/            # Formant plots
│   ├── specs/            # Spectrograms
│   ├── pitch_plots/      # Pitch analysis plots
│   └── synth_outputs/    # Synthetic audio & comparisons
│       └── comparisons/  # Comparison visualizations
├── templates/            # HTML templates (Jinja2)
│   ├── index.html
│   ├── results.html
│   ├── pitch_report.html
│   └── synthesis.html
└── utils/
    ├── analysis.py       # Formant/pitch extraction
    ├── plotting.py       # Visualization functions
    └── file_utils.py     # File handling utilities
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/HusenAbugosh/SpokenA1.git
cd SpokenA1
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Required libraries**:
- Flask
- pandas
- numpy
- matplotlib
- parselmouth-praat
- librosa
- soundfile
- scipy

## Usage

### Running the Web Application

Start the Flask server:
```bash
python app.py
```

Access the application at: `http://127.0.0.1:5000`

### Navigation

- **Home**: Overview and dataset information
- **Vowel Analysis**: View F1/F2/F3 formants, duration, and spectrograms
- **Pitch Analysis**: Compare Praat vs Librosa pitch tracking methods
- **Synthesis**: Generate synthetic vowels and view detailed comparisons

### Standalone Comparison Script

Generate all comparisons without running the web server:
```bash
python run_comparison.py
```

This will:
- Synthesize all 5 vowels
- Compare each with 2 natural samples
- Save plots to `static/synth_outputs/comparisons/`
- Export statistics to `comparison_summary.csv`

## Technical Details

### Synthesis Parameters
- **Sampling Rate**: 16,000 Hz
- **Duration**: 0.5 seconds per vowel
- **Fundamental Frequency**: 120 Hz
- **Source**: Impulse train (glottal pulses)
- **Filter**: Cascaded 2nd-order IIR resonators (one per formant)

### Formant Extraction
- Method: Linear Predictive Coding (LPC) via Praat
- Window: Gaussian (25 ms)
- Pre-emphasis: Applied
- Maximum formant frequency: 5500 Hz

### Comparison Metrics
- **F1 Error**: Absolute difference in Hz between synthetic and natural F1
- **F2 Error**: Absolute difference in Hz between synthetic and natural F2
- **Average Errors** (Current Results):
  - F1: 148.1 Hz
  - F2: 590.5 Hz

## Results Summary

| Vowel | Best F1 Match | Best F2 Match | Characteristics |
|-------|---------------|---------------|-----------------|
| /i/   | High accuracy | Front vowel   | High F2 (~2300 Hz) |
| /e/   | Mid-high      | Front-mid     | F2 ~1900 Hz |
| /a/   | Low F1        | Central       | Most open vowel |
| /o/   | **Best** (13.2 Hz error) | Back-mid | Rounded |
| /u/   | High back     | Low F2        | Most rounded |

## Known Limitations

1. **Robotic Sound Quality**: The synthetic vowels sound robotic because:
   - Simple impulse train source (no natural glottal waveform)
   - Fixed F0 (no prosodic variation)
   - No formant transitions
   - Lacks natural breathiness and aspiration

2. **Formant Accuracy**: F2 errors are larger than F1 errors due to:
   - Higher sensitivity to speaker variation
   - More challenging to synthesize accurately
   - Greater articulatory complexity

## Future Enhancements

- [ ] More natural excitation source (e.g., Klatt synthesizer)
- [ ] Formant transition modeling
- [ ] Prosodic variation (F0 contours)
- [ ] Additional vowel quality metrics (spectral tilt, bandwidth)
- [ ] Real-time synthesis interface



## Contributors

Developed by Husen Abugosh  and kareem qutob for the Spoken Language Processing course 



## Acknowledgments

- Praat/Parselmouth for acoustic analysis tools
- Librosa for audio processing utilities
- Flask framework for web application structure
