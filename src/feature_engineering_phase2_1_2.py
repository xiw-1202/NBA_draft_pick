"""
Feature Engineering - Phase 2.1: Basic Statistical Features
Step 2.1.2: Calculate per-36-minute statistics

This script:
- Flags players with < 100 total minutes (keeps all but marks them)
- Calculates per-36-minute stats to normalize for playing time
- Caps outliers at 99th percentile (only for players with enough minutes)
- Sets per-36 stats to 0 for low-minute players
- Creates standardized per-36 features for model input
"""

import pandas as pd
import numpy as np
import os

def calculate_per_36_stats(input_file, output_file):
    """
    Calculate per-36-minute statistics for all counting stats.

    Per-36 stats normalize player performance to a standard 36 minutes,
    allowing fair comparison between starters and bench players.

    Args:
        input_file: Path to dataset with per-game features
        output_file: Path to save dataset with per-36 features

    Returns:
        DataFrame with calculated per-36 statistics
    """
    print("="*80)
    print("Phase 2.1.2: Calculating Per-36-Minute Statistics")
    print("="*80)

    # Load data
    print("\n[1/6] Loading feature dataset...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"   → Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Calculate total minutes
    print("\n[2/6] Calculating total minutes played...")
    df['total_minutes'] = df['MPG'] * df['GP']
    print(f"   → Total minutes range: {df['total_minutes'].min():.1f} to {df['total_minutes'].max():.1f}")
    print(f"   → Median total minutes: {df['total_minutes'].median():.1f}")

    # Filter players with < 100 total minutes
    print("\n[3/6] Filtering players by minimum playing time...")
    low_minutes = df['total_minutes'] < 100
    print(f"   → Players with < 100 total minutes: {low_minutes.sum():,}")
    print(f"   → Players with >= 100 total minutes: {(~low_minutes).sum():,}")

    # Create filtered dataframe but keep all for now (mark with flag)
    df['enough_minutes'] = ~low_minutes
    print(f"   ℹ️  Keeping all players but flagging low-minute players")

    # Calculate per-36 statistics
    print("\n[4/6] Calculating per-36 statistics...")

    # Avoid division by zero
    mpg_safe = df['MPG'].replace(0, np.nan)

    # Calculate per-36 for counting stats
    per_36_stats = {
        'PTS_36': df['PPG'],
        'REB_36': df['RPG'],
        'OREB_36': df['OREB_PG'],
        'DREB_36': df['DREB_PG'],
        'AST_36': df['APG'],
        'STL_36': df['SPG'],
        'BLK_36': df['BPG'],
        'FTM_36': df['FTM_PG'],
        'FTA_36': df['FTA_PG'],
        'TPM_36': df['TPM_PG'],
        'TPA_36': df['TPA_PG'],
        '2PM_36': df['twoPM_PG'],
        '2PA_36': df['twoPA_PG'],
    }

    for col_name, per_game_stat in per_36_stats.items():
        df[col_name] = (per_game_stat / mpg_safe) * 36
        df[col_name] = df[col_name].fillna(0)  # Fill NaN from division by zero

    print("   ✅ Created 13 per-36 features")
    per_36_cols = list(per_36_stats.keys())

    # Cap outliers at 99th percentile
    print("\n[5/6] Capping outliers at 99th percentile...")
    capped_count = 0

    for col in per_36_cols:
        # Only cap for players with enough minutes
        valid_mask = df['enough_minutes']
        percentile_99 = df.loc[valid_mask, col].quantile(0.99)

        outliers = (df[col] > percentile_99) & valid_mask
        outlier_count = outliers.sum()

        if outlier_count > 0:
            print(f"   → {col}: capping {outlier_count} values at {percentile_99:.2f}")
            df.loc[outliers, col] = percentile_99
            capped_count += outlier_count

    print(f"   ✅ Capped {capped_count} total outlier values")

    # Set per-36 stats to 0 for low-minute players
    print("\n[6/6] Handling low-minute players...")
    low_minute_mask = ~df['enough_minutes']
    for col in per_36_cols:
        df.loc[low_minute_mask, col] = 0
    print(f"   ✅ Set per-36 stats to 0 for {low_minute_mask.sum():,} low-minute players")

    # Validation
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)

    validation_results = []

    # Check for remaining NaN/inf
    for col in per_36_cols:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        if nan_count > 0 or inf_count > 0:
            validation_results.append(f"   ⚠️  {col}: {nan_count} NaN, {inf_count} inf values")

    # Check for negative values
    for col in per_36_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            validation_results.append(f"   ⚠️  {col} has {neg_count} negative values")

    if len(validation_results) == 0:
        print("   ✅ All validation checks passed")
    else:
        for result in validation_results:
            print(result)

    # Summary statistics (only for players with enough minutes)
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Players with >= 100 minutes)")
    print("="*80)
    valid_players = df[df['enough_minutes']]
    summary_stats = valid_players[per_36_cols].describe().round(2)
    print(summary_stats)

    print(f"\nPlayers included in summary: {len(valid_players):,}")

    # Top performers for verification
    print("\n" + "="*80)
    print("TOP 10 SCORERS (Per-36-Minute Basis)")
    print("="*80)
    top_scorers_36 = df[df['enough_minutes']].nlargest(10, 'PTS_36')[
        ['Player', 'Team', 'Season', 'MPG', 'PPG', 'PTS_36', 'total_minutes']
    ]
    print(top_scorers_36.to_string(index=False))

    # Compare per-game vs per-36 leaders
    print("\n" + "="*80)
    print("INTERESTING COMPARISONS: Per-Game vs Per-36")
    print("="*80)

    # Find players who benefit most from per-36 (low MPG, high efficiency)
    efficiency_subset = df[df['enough_minutes'] & (df['MPG'] < 25)].nlargest(5, 'PTS_36').copy()
    efficiency_subset['pts_36_boost'] = efficiency_subset['PTS_36'] - efficiency_subset['PPG']
    efficiency_players = efficiency_subset[
        ['Player', 'Team', 'Season', 'MPG', 'PPG', 'PTS_36', 'pts_36_boost']
    ]
    print("\nEfficient Low-Minute Players (MPG < 25):")
    print(efficiency_players.to_string(index=False))

    # Save output
    print("\n" + "="*80)
    print("SAVING OUTPUT")
    print("="*80)
    df.to_csv(output_file, index=False)
    print(f"   ✅ Saved to: {output_file}")
    print(f"   → Shape: {df.shape}")
    print(f"   → New columns added: {len(per_36_cols) + 2}")  # +2 for total_minutes and enough_minutes
    print(f"   → Players with enough minutes: {df['enough_minutes'].sum():,}")

    return df


def main():
    """Main execution function"""

    # Define paths
    base_dir = "/Users/sam/Documents/School/Emory/DataLab"
    input_file = os.path.join(base_dir, "data/final/features_phase2_1_1.csv")
    output_file = os.path.join(base_dir, "data/final/features_phase2_1_2.csv")

    # Check input exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Run calculation
    df_with_features = calculate_per_36_stats(input_file, output_file)

    print("\n" + "="*80)
    print("✅ PHASE 2.1.2 COMPLETE")
    print("="*80)
    print(f"Output saved to: {output_file}")
    print(f"Ready for Phase 2.1.3: Calculate shooting percentages")


if __name__ == "__main__":
    main()
