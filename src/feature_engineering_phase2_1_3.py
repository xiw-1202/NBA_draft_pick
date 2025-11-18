"""
Feature Engineering - Phase 2.1: Basic Statistical Features
Step 2.1.3: Calculate shooting percentages

This script:
- Standardizes all shooting percentages to decimal format (0-1 range)
- Calculates overall FG% (field goal percentage)
- Validates all percentages are in proper range
- Handles division by zero cases (0 attempts)
"""

import pandas as pd
import numpy as np
import os

def calculate_shooting_percentages(input_file, output_file):
    """
    Calculate and standardize all shooting percentage statistics.

    Shooting percentages are crucial for evaluating player efficiency:
    - FG%: Overall field goal percentage
    - 2P%: Two-point shooting percentage
    - 3P%: Three-point shooting percentage
    - FT%: Free throw percentage
    - eFG%: Effective field goal % (weights 3P more heavily)
    - TS%: True shooting % (accounts for FT, 2P, 3P)

    Args:
        input_file: Path to dataset with per-36 features
        output_file: Path to save dataset with shooting percentages

    Returns:
        DataFrame with calculated and standardized shooting percentages
    """
    print("="*80)
    print("Phase 2.1.3: Calculating Shooting Percentages")
    print("="*80)

    # Load data
    print("\n[1/5] Loading feature dataset...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"   → Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Calculate total field goals (made and attempted)
    print("\n[2/5] Calculating total field goal statistics...")

    # Total FG = 2PM + 3PM
    df['FGM'] = df['twoPM'] + df['TPM']
    df['FGA'] = df['twoPA'] + df['TPA']

    print(f"   → FGM range: {df['FGM'].min():.0f} to {df['FGM'].max():.0f}")
    print(f"   → FGA range: {df['FGA'].min():.0f} to {df['FGA'].max():.0f}")
    print("   ✅ Created FGM and FGA columns")

    # Calculate shooting percentages
    print("\n[3/5] Calculating and standardizing shooting percentages...")

    # Overall FG% (new calculation)
    df['FG_PCT'] = np.where(df['FGA'] > 0, df['FGM'] / df['FGA'], 0)
    print("   ✅ Calculated FG_PCT (overall field goal %)")

    # 2P% - already exists as twoP_per, verify and rename
    df['FG2_PCT'] = np.where(df['twoPA'] > 0, df['twoPM'] / df['twoPA'], 0)
    matches_existing = np.allclose(df['FG2_PCT'].fillna(0), df['twoP_per'].fillna(0), rtol=0.01)
    print(f"   ✅ Calculated FG2_PCT (2-point %) - matches twoP_per: {matches_existing}")

    # 3P% - already exists as TP_per, verify and rename
    df['FG3_PCT'] = np.where(df['TPA'] > 0, df['TPM'] / df['TPA'], 0)
    matches_existing = np.allclose(df['FG3_PCT'].fillna(0), df['TP_per'].fillna(0), rtol=0.01)
    print(f"   ✅ Calculated FG3_PCT (3-point %) - matches TP_per: {matches_existing}")

    # FT% - already exists as FT_per, verify
    df['FT_PCT'] = np.where(df['FTA'] > 0, df['FTM'] / df['FTA'], 0)
    matches_existing = np.allclose(df['FT_PCT'].fillna(0), df['FT_per'].fillna(0), rtol=0.01)
    print(f"   ✅ Calculated FT_PCT (free throw %) - matches FT_per: {matches_existing}")

    # Standardize eFG% (currently in 0-100 range, convert to 0-1)
    df['eFG_PCT'] = df['eFG'] / 100
    # Cap at 1.0 (some edge cases have values > 100%)
    over_one = (df['eFG_PCT'] > 1.0).sum()
    if over_one > 0:
        print(f"   ℹ️  Capping {over_one} eFG_PCT values > 1.0")
        df['eFG_PCT'] = df['eFG_PCT'].clip(upper=1.0)
    print("   ✅ Standardized eFG_PCT (effective FG%) to 0-1 range")

    # Standardize TS% (currently in 0-100 range, convert to 0-1)
    df['TS_PCT'] = df['TS_per'] / 100
    # Cap at 1.0 (some edge cases have values > 100%)
    over_one = (df['TS_PCT'] > 1.0).sum()
    if over_one > 0:
        print(f"   ℹ️  Capping {over_one} TS_PCT values > 1.0")
        df['TS_PCT'] = df['TS_PCT'].clip(upper=1.0)
    print("   ✅ Standardized TS_PCT (true shooting %) to 0-1 range")

    # Validate percentages are in 0-1 range
    print("\n[4/5] Validating shooting percentages...")

    pct_cols = ['FG_PCT', 'FG2_PCT', 'FG3_PCT', 'FT_PCT', 'eFG_PCT', 'TS_PCT']
    validation_results = []

    for col in pct_cols:
        # Check for values outside 0-1 range
        out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
        if out_of_range > 0:
            validation_results.append(f"   ⚠️  {col}: {out_of_range} values outside [0, 1] range")

        # Check for NaN
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            validation_results.append(f"   ⚠️  {col}: {nan_count} NaN values")

        # Check for inf
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            validation_results.append(f"   ⚠️  {col}: {inf_count} inf values")

    if len(validation_results) == 0:
        print("   ✅ All validation checks passed")
    else:
        for result in validation_results:
            print(result)

    # Summary statistics
    print("\n[5/5] Generating summary statistics...")
    print("\n" + "="*80)
    print("SHOOTING PERCENTAGE SUMMARY")
    print("="*80)

    summary_stats = df[pct_cols].describe().round(3)
    print(summary_stats)

    # Identify elite shooters
    print("\n" + "="*80)
    print("ELITE SHOOTERS (Minimum 100 total minutes)")
    print("="*80)

    qualified = df[df['enough_minutes']].copy()

    print("\nTop 10 Overall FG% (min 100 FGA):")
    top_fg = qualified[qualified['FGA'] >= 100].nlargest(10, 'FG_PCT')[
        ['Player', 'Team', 'Season', 'FGA', 'FGM', 'FG_PCT']
    ]
    top_fg['FG_PCT'] = (top_fg['FG_PCT'] * 100).round(1).astype(str) + '%'
    print(top_fg.to_string(index=False))

    print("\nTop 10 Three-Point % (min 50 3PA):")
    top_3p = qualified[qualified['TPA'] >= 50].nlargest(10, 'FG3_PCT')[
        ['Player', 'Team', 'Season', 'TPA', 'TPM', 'FG3_PCT']
    ]
    top_3p['FG3_PCT'] = (top_3p['FG3_PCT'] * 100).round(1).astype(str) + '%'
    print(top_3p.to_string(index=False))

    print("\nTop 10 Free Throw % (min 50 FTA):")
    top_ft = qualified[qualified['FTA'] >= 50].nlargest(10, 'FT_PCT')[
        ['Player', 'Team', 'Season', 'FTA', 'FTM', 'FT_PCT']
    ]
    top_ft['FT_PCT'] = (top_ft['FT_PCT'] * 100).round(1).astype(str) + '%'
    print(top_ft.to_string(index=False))

    # Interesting comparison: eFG% vs FG%
    print("\n" + "="*80)
    print("SHOOTING EFFICIENCY ANALYSIS")
    print("="*80)

    qualified_shooters = qualified[qualified['FGA'] >= 100].copy()
    qualified_shooters['eFG_boost'] = qualified_shooters['eFG_PCT'] - qualified_shooters['FG_PCT']

    print("\nPlayers with biggest eFG% boost (3-point volume shooters):")
    top_boost = qualified_shooters.nlargest(5, 'eFG_boost')[
        ['Player', 'Team', 'Season', 'FG_PCT', 'eFG_PCT', 'eFG_boost', 'FG3_PCT']
    ]
    top_boost['FG_PCT'] = (top_boost['FG_PCT'] * 100).round(1).astype(str) + '%'
    top_boost['eFG_PCT'] = (top_boost['eFG_PCT'] * 100).round(1).astype(str) + '%'
    top_boost['eFG_boost'] = (top_boost['eFG_boost'] * 100).round(1).astype(str) + '%'
    top_boost['FG3_PCT'] = (top_boost['FG3_PCT'] * 100).round(1).astype(str) + '%'
    print(top_boost.to_string(index=False))

    # Save output
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    df.to_csv(output_file, index=False)
    print(f"   ✅ Saved to: {output_file}")
    print(f"   → Shape: {df.shape}")
    print(f"   → New columns added: {len(pct_cols) + 2}")  # +2 for FGM and FGA
    print(f"   → Shooting percentage columns: {', '.join(pct_cols)}")

    return df


def main():
    """Main execution function"""

    # Define paths
    base_dir = "/Users/sam/Documents/School/Emory/DataLab"
    input_file = os.path.join(base_dir, "data/final/features_phase2_1_2.csv")
    output_file = os.path.join(base_dir, "data/final/features_phase2_1_3.csv")

    # Check input exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Run calculation
    df_with_features = calculate_shooting_percentages(input_file, output_file)

    print("\n" + "="*80)
    print("✅ PHASE 2.1.3 COMPLETE")
    print("="*80)
    print(f"Output saved to: {output_file}")
    print(f"Ready for Phase 2.2: Advanced Basketball Metrics")


if __name__ == "__main__":
    main()
