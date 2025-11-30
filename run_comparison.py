"""
Standalone script to run detailed vowel comparison between synthetic and natural vowels.
This compares each of the 5 synthetic vowels with 2 natural samples.

Usage:
    python run_comparison.py
"""

import os
import pandas as pd
from synth import batch_synthesize, detailed_vowel_comparison

# Configuration
DATA_PATH = "Data"  # Path to your natural vowel recordings
OUTPUT_DIR = "static/synth_outputs"
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")

def main():
    print("=" * 60)
    print("VOWEL SYNTHESIS & COMPARISON")
    print("=" * 60)
    
    # Step 1: Synthesize all 5 vowels
    print("\n[1/3] Synthesizing vowels (i, e, a, o, u)...")
    synth_rows = batch_synthesize(output_dir=OUTPUT_DIR)
    print(f"✓ Generated {len(synth_rows)} synthetic vowels")
    
    # Display synthetic vowel formants
    print("\nSynthetic Vowel Formants:")
    for row in synth_rows:
        print(f"  /{row['Vowel']}/ → F1={row['F1']:.1f} Hz, F2={row['F2']:.1f} Hz")
    
    # Step 2: Run detailed comparison with natural samples
    print(f"\n[2/3] Comparing with natural samples from '{DATA_PATH}'...")
    
    if not os.path.isdir(DATA_PATH):
        print(f"✗ Error: Data folder '{DATA_PATH}' not found!")
        print("  Please ensure your vowel recordings are in the Data/ folder")
        return
    
    comparison_results, summary_path = detailed_vowel_comparison(
        DATA_PATH, 
        synth_rows, 
        output_dir=COMPARISON_DIR
    )
    
    print(f"✓ Created {len(comparison_results)} detailed comparison plots")
    print(f"✓ Summary plot saved: {summary_path}")
    
    # Step 3: Display results
    print("\n[3/3] Comparison Results:")
    print("-" * 60)
    
    for result in comparison_results:
        print(f"\nVowel /{result['Vowel']}/:")
        print(f"  Synthetic:  F1={result['Synth_F1']:.1f} Hz, F2={result['Synth_F2']:.1f} Hz")
        print(f"  Natural 1:  F1={result['Natural1_F1']:.1f} Hz, F2={result['Natural1_F2']:.1f} Hz  ({result['Natural1_File']})")
        print(f"  Natural 2:  F1={result['Natural2_F1']:.1f} Hz, F2={result['Natural2_F2']:.1f} Hz  ({result['Natural2_File']})")
        print(f"  Avg Error:  ΔF1={result['F1_Error']:.1f} Hz, ΔF2={result['F2_Error']:.1f} Hz")
        print(f"  Plot: {result['ComparisonPlot']}")
    
    # Create summary CSV
    csv_path = os.path.join(COMPARISON_DIR, "comparison_summary.csv")
    df = pd.DataFrame(comparison_results)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary CSV saved: {csv_path}")
    
    # Calculate overall statistics
    avg_f1_error = df['F1_Error'].mean()
    avg_f2_error = df['F2_Error'].mean()
    
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Average F1 Error: {avg_f1_error:.1f} Hz")
    print(f"Average F2 Error: {avg_f2_error:.1f} Hz")
    print(f"\nAll comparison plots saved in: {COMPARISON_DIR}/")
    print("=" * 60)
    
    print("\n✓ Done! You can now:")
    print("  1. View individual comparison plots in:", COMPARISON_DIR)
    print("  2. View summary plot:", summary_path)
    print("  3. View detailed results in:", csv_path)
    print("  4. Or run the Flask app: python app.py")

if __name__ == "__main__":
    main()
