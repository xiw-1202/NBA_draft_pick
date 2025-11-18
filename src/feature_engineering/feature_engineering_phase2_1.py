"""
Feature Engineering - Phase 2.1: Basic Statistical Features
Step 2.1.1: Calculate per-game averages

This script:
- Verifies existing per-game statistics
- Creates standardized column names (PPG, RPG, APG, etc.)
- Handles division by zero cases
- Validates calculations
"""

import pandas as pd
import numpy as np
import os

def calculate_per_game_averages(input_file, output_file):
    """
    Calculate and verify per-game averages for all basic statistics.

    Args:
        input_file: Path to merged dataset
        output_file: Path to save dataset with per-game features

    Returns:
        DataFrame with calculated per-game statistics
    """
    print("="*80)
    print("Phase 2.1.1: Calculating Per-Game Averages")
    print("="*80)

    # Load data
    print("\n[1/6] Loading merged dataset...")
    df = pd.read_csv(input_file)
    print(f"   → Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Check games played
    print("\n[2/6] Checking GP (Games Played) column...")
    print(f"   → GP range: {df['GP'].min()} to {df['GP'].max()}")
    print(f"   → Players with GP = 0: {(df['GP'] == 0).sum()}")
    print(f"   → Players with GP > 0: {(df['GP'] > 0).sum()}")

    # Filter out players with 0 games (if any)
    if (df['GP'] == 0).sum() > 0:
        print(f"   ⚠️  Removing {(df['GP'] == 0).sum()} players with GP = 0")
        df = df[df['GP'] > 0].copy()

    # Verify existing per-game stats
    print("\n[3/6] Verifying existing per-game statistics...")

    # Check if pts is already per-game by examining value magnitudes
    # If pts is total: should have many values > 100 (season totals)
    # If pts is per-game: should have most values < 50 (per-game stats)
    pts_median = df['pts'].median()
    pts_95th = df['pts'].quantile(0.95)
    high_values_pct = (df['pts'] > 100).sum() / len(df) * 100

    print(f"   → pts median: {pts_median:.2f}")
    print(f"   → pts 95th percentile: {pts_95th:.2f}")
    print(f"   → % of values > 100: {high_values_pct:.2f}%")

    if pts_95th < 50:
        print("   ✅ Stats appear to already be in per-game format")
        stats_are_per_game = True
    else:
        print("   ℹ️  Stats appear to be totals, will calculate per-game")
        stats_are_per_game = False

    # Create per-game columns with standard names
    print("\n[4/6] Creating standardized per-game columns...")

    if stats_are_per_game:
        # Stats are already per-game, just rename
        df['PPG'] = df['pts']
        df['RPG'] = df['treb']  # total rebounds
        df['OREB_PG'] = df['oreb']  # offensive rebounds
        df['DREB_PG'] = df['dreb']  # defensive rebounds
        df['APG'] = df['ast']
        df['SPG'] = df['stl']
        df['BPG'] = df['blk']
        df['MPG'] = df['Min_per']  # minutes per game
        df['FTM_PG'] = df['FTM'] / df['GP']  # free throws made (these might be totals)
        df['FTA_PG'] = df['FTA'] / df['GP']  # free throws attempted
        df['TPM_PG'] = df['TPM'] / df['GP']  # three pointers made
        df['TPA_PG'] = df['TPA'] / df['GP']  # three pointers attempted
        df['twoPM_PG'] = df['twoPM'] / df['GP']  # two pointers made
        df['twoPA_PG'] = df['twoPA'] / df['GP']  # two pointers attempted

        print("   ✅ Created PPG, RPG, APG, SPG, BPG, MPG")
        print("   ✅ Calculated FTM_PG, FTA_PG, TPM_PG, TPA_PG from totals")
    else:
        # Calculate per-game from totals
        df['PPG'] = df['pts'] / df['GP']
        df['RPG'] = df['treb'] / df['GP']
        df['OREB_PG'] = df['oreb'] / df['GP']
        df['DREB_PG'] = df['dreb'] / df['GP']
        df['APG'] = df['ast'] / df['GP']
        df['SPG'] = df['stl'] / df['GP']
        df['BPG'] = df['blk'] / df['GP']
        df['MPG'] = df['Min_per']
        df['FTM_PG'] = df['FTM'] / df['GP']
        df['FTA_PG'] = df['FTA'] / df['GP']
        df['TPM_PG'] = df['TPM'] / df['GP']
        df['TPA_PG'] = df['TPA'] / df['GP']
        df['twoPM_PG'] = df['twoPM'] / df['GP']
        df['twoPA_PG'] = df['twoPA'] / df['GP']

        print("   ✅ Calculated all per-game statistics from totals")

    # Handle any NaN or infinite values
    print("\n[5/6] Handling missing/invalid values...")
    per_game_cols = ['PPG', 'RPG', 'OREB_PG', 'DREB_PG', 'APG', 'SPG', 'BPG',
                     'MPG', 'FTM_PG', 'FTA_PG', 'TPM_PG', 'TPA_PG', 'twoPM_PG', 'twoPA_PG']

    for col in per_game_cols:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()

        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️  {col}: {nan_count} NaN, {inf_count} inf values")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0)

    # Validation
    print("\n[6/6] Validation checks...")
    validation_results = []

    # Check for negative values
    for col in per_game_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            validation_results.append(f"   ⚠️  {col} has {neg_count} negative values")

    # Check for reasonable ranges
    if df['PPG'].max() > 50:
        validation_results.append(f"   ⚠️  Max PPG is {df['PPG'].max():.2f} (seems high)")

    if df['MPG'].max() > 45:
        validation_results.append(f"   ⚠️  Max MPG is {df['MPG'].max():.2f} (>45 minutes)")

    if len(validation_results) == 0:
        print("   ✅ All validation checks passed")
    else:
        for result in validation_results:
            print(result)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary_stats = df[per_game_cols].describe().round(2)
    print(summary_stats)

    # Top scorers for verification
    print("\n" + "="*80)
    print("TOP 10 SCORERS (Verification)")
    print("="*80)
    top_scorers = df.nlargest(10, 'PPG')[['Player', 'Team', 'Season', 'GP', 'PPG', 'RPG', 'APG']]
    print(top_scorers.to_string(index=False))

    # Save output
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    df.to_csv(output_file, index=False)
    print(f"   ✅ Saved to: {output_file}")
    print(f"   → Shape: {df.shape}")
    print(f"   → New columns added: {len(per_game_cols)}")

    return df


def main():
    """Main execution function"""

    # Define paths
    base_dir = "/Users/sam/Documents/School/Emory/DataLab"
    input_file = os.path.join(base_dir, "data/final/merged_dataset_final.csv")
    output_file = os.path.join(base_dir, "data/final/features_phase2_1_1.csv")

    # Check input exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Run calculation
    df_with_features = calculate_per_game_averages(input_file, output_file)

    print("\n" + "="*80)
    print("✅ PHASE 2.1.1 COMPLETE")
    print("="*80)
    print(f"Output saved to: {output_file}")
    print(f"Ready for Phase 2.1.2: Calculate per-36-minute statistics")


if __name__ == "__main__":
    main()
