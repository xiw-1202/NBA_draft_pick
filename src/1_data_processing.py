"""
NBA Draft Prediction - Data Processing (Hybrid Approach)
=========================================================
Strategy: Use final college year per player + calculate improvement features

Author: Sam
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "RAW"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "PROCESSED"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DATA PROCESSING - HYBRID APPROACH")
print("Final Year + Improvement Features")
print("=" * 80)

# ============================================================================
# 1. Load College Data
# ============================================================================
print("\n1️⃣ Loading college basketball data...")

df_college = pd.read_csv(DATA_DIR / "CollegeBasketballPlayers2009-2021.csv")
print(f"   Raw data: {df_college.shape}")
print(f"   Years covered: {df_college['year'].min()} - {df_college['year'].max()}")

# Clean player names
df_college["player_name_clean"] = df_college["player_name"].str.strip()

# ============================================================================
# 2. Identify Each Player's College Career
# ============================================================================
print("\n2️⃣ Analyzing player careers...")

# Group by player to understand their college timeline
player_career = (
    df_college.groupby("player_name_clean")
    .agg(
        {
            "year": ["min", "max", "count"],
            "yr": lambda x: (
                x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1]
            ),  # Most common class
            "pick": "max",  # If drafted, will have a pick number
        }
    )
    .reset_index()
)

player_career.columns = [
    "player_name_clean",
    "first_year",
    "last_year",
    "years_played",
    "typical_class",
    "draft_pick",
]

print(f"   Total unique players: {len(player_career):,}")
print(f"   Players with 1 year: {(player_career['years_played'] == 1).sum():,}")
print(f"   Players with 2 years: {(player_career['years_played'] == 2).sum():,}")
print(f"   Players with 3 years: {(player_career['years_played'] == 3).sum():,}")
print(f"   Players with 4+ years: {(player_career['years_played'] >= 4).sum():,}")
print(f"   Players drafted: {player_career['draft_pick'].notna().sum():,}")

# ============================================================================
# 3. Calculate Career Statistics (for improvement features)
# ============================================================================
print("\n3️⃣ Calculating career statistics and improvement...")

# Sort by player and year
df_sorted = df_college.sort_values(["player_name_clean", "year"])

# For each player, calculate career stats up to each year
print("   Computing year-over-year changes...")

# Key stats to track improvement
improvement_stats = ["pts", "treb", "ast", "TS_per", "bpm", "Min_per", "usg"]

for stat in improvement_stats:
    if stat in df_sorted.columns:
        # Previous year value
        df_sorted[f"{stat}_prev"] = df_sorted.groupby("player_name_clean")[stat].shift(
            1
        )

        # Improvement from previous year
        df_sorted[f"{stat}_improvement"] = df_sorted[stat] - df_sorted[f"{stat}_prev"]

        # First year value (for total improvement calculation)
        df_sorted[f"{stat}_first_year"] = df_sorted.groupby("player_name_clean")[
            stat
        ].transform("first")

        # Total improvement from freshman year
        df_sorted[f"{stat}_total_improvement"] = (
            df_sorted[stat] - df_sorted[f"{stat}_first_year"]
        )

# Career averages (up to current year)
print("   Computing career averages...")
for stat in improvement_stats:
    if stat in df_sorted.columns:
        df_sorted[f"{stat}_career_avg"] = df_sorted.groupby("player_name_clean")[
            stat
        ].transform("mean")

# Years in college (at each point)
df_sorted["years_in_college"] = df_sorted.groupby("player_name_clean").cumcount() + 1

print("   ✅ Improvement features calculated")

# ============================================================================
# 4. Identify Final Year for Each Player
# ============================================================================
print("\n4️⃣ Filtering to final college year per player...")

# Strategy: For each player, take their LAST year in college
# Special case: If they have 'pick' value, that's their draft year - use year before


def get_final_year_mask(group):
    """For each player group, mark their final college year"""
    # If player was drafted, their draft year should have 'pick' value
    # We want the last year they played
    mask = group["year"] == group["year"].max()
    return mask


# Apply to get final year for each player
final_year_mask = df_sorted.groupby("player_name_clean", group_keys=False).apply(
    lambda g: g["year"] == g["year"].max()
)

df_final_year = df_sorted[final_year_mask].copy()

print(f"   Original dataset: {len(df_sorted):,} rows")
print(f"   Final year only: {len(df_final_year):,} rows")
print(f"   Unique players: {df_final_year['player_name_clean'].nunique():,}")

# Verify we have one row per player
duplicates = df_final_year["player_name_clean"].duplicated().sum()
if duplicates > 0:
    print(f"   ⚠️  Warning: {duplicates} duplicate players found")
    # Keep last occurrence (most recent year)
    df_final_year = df_final_year.drop_duplicates(
        subset="player_name_clean", keep="last"
    )
    print(f"   ✅ Deduplicated to: {len(df_final_year):,} rows")

# ============================================================================
# 5. Load and Clean Draft Data
# ============================================================================
print("\n5️⃣ Loading draft data...")

df_drafted = pd.read_excel(DATA_DIR / "DraftedPlayers2009-2021.xlsx")

# Remove header row if present
if df_drafted.iloc[0]["ROUND"] == "NUMBER":
    df_drafted = df_drafted.iloc[1:].reset_index(drop=True)

# Clean data
df_drafted["YEAR"] = pd.to_numeric(df_drafted["YEAR"], errors="coerce")
df_drafted["PICK"] = pd.to_numeric(df_drafted["OVERALL"], errors="coerce")
df_drafted["ROUND"] = pd.to_numeric(df_drafted["ROUND"], errors="coerce")

# Clean player names
df_drafted["player_name_clean"] = df_drafted["PLAYER"].str.strip()

# Remove invalid rows
df_drafted = df_drafted.dropna(subset=["player_name_clean", "YEAR", "PICK"])

print(f"   Drafted players: {len(df_drafted):,}")
print(
    f"   Draft years: {df_drafted['YEAR'].min():.0f} - {df_drafted['YEAR'].max():.0f}"
)

# ============================================================================
# 6. Merge College + Draft Data
# ============================================================================
print("\n6️⃣ Merging college stats with draft data...")


# Standardize names for matching
def standardize_name(name):
    """Standardize player names for better matching"""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    # Remove suffixes
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    # Remove punctuation except spaces
    name = re.sub(r"[^\w\s]", "", name)
    # Normalize spaces
    name = re.sub(r"\s+", " ", name).strip()
    return name


df_final_year["name_std"] = df_final_year["player_name_clean"].apply(standardize_name)
df_drafted["name_std"] = df_drafted["player_name_clean"].apply(standardize_name)

# Create a draft lookup with standardized names
draft_lookup = df_drafted[["name_std", "YEAR", "PICK", "ROUND"]].rename(
    columns={"YEAR": "draft_year", "PICK": "draft_pick", "ROUND": "draft_round"}
)

# Merge strategy: Match on name
# Note: draft_year is usually college_year + 1, but we'll match by name only
df_merged = df_final_year.merge(draft_lookup, on="name_std", how="left")

# Create binary target: was_drafted
df_merged["was_drafted"] = df_merged["draft_pick"].notna().astype(int)

print(f"   Merged dataset: {len(df_merged):,} players")
print(f"   Drafted players matched: {df_merged['was_drafted'].sum():,}")
print(f"   Undrafted players: {(df_merged['was_drafted'] == 0).sum():,}")

# ============================================================================
# 7. Load and Merge RAPTOR (NBA Performance) Data
# ============================================================================
print("\n7️⃣ Loading NBA performance data (RAPTOR)...")

df_raptor_modern = pd.read_csv(DATA_DIR / "modern_RAPTOR_by_player.csv")
df_raptor_latest = pd.read_csv(DATA_DIR / "latest_RAPTOR_by_player.csv")

# Combine RAPTOR data
df_raptor = pd.concat([df_raptor_modern, df_raptor_latest], ignore_index=True)
print(f"   Total RAPTOR records: {len(df_raptor):,}")

# Aggregate by player (career stats)
raptor_agg = (
    df_raptor.groupby("player_name")
    .agg(
        {
            "season": ["min", "max", "count"],
            "poss": "sum",
            "mp": "sum",
            "raptor_offense": "mean",
            "raptor_defense": "mean",
            "raptor_total": "mean",
            "war_total": "sum",
            "war_reg_season": "sum",
        }
    )
    .reset_index()
)

# Flatten columns
raptor_agg.columns = [
    "_".join(col).strip("_") if col[1] else col[0] for col in raptor_agg.columns.values
]
raptor_agg.rename(columns={"player_name": "player_name_clean"}, inplace=True)

# Standardize names
raptor_agg["name_std"] = raptor_agg["player_name_clean"].apply(standardize_name)

print(f"   Unique NBA players: {len(raptor_agg):,}")

# Merge with our dataset
df_merged = df_merged.merge(
    raptor_agg[
        ["name_std", "raptor_total_mean", "war_total_sum", "season_count", "mp_sum"]
    ],
    on="name_std",
    how="left",
)

nba_players = df_merged["raptor_total_mean"].notna().sum()
print(f"   Players with NBA data: {nba_players:,}")

# ============================================================================
# 8. Create Target Variables
# ============================================================================
print("\n8️⃣ Creating target variables...")

# 1. Binary: Was drafted
df_merged["target_drafted"] = df_merged["was_drafted"]

# 2. Continuous: Draft pick (1-60, NaN for undrafted)
df_merged["target_draft_pick"] = df_merged["draft_pick"]

# 3. Diamond player indicator
# Definition: Drafted late (pick > 30) OR undrafted, BUT had successful NBA career
# Success criteria:
#   - Career WAR >= 10.0 (significant positive contribution)
#   - Minutes >= 2000 (meaningful playing time, ~1-2 seasons of regular minutes)
# This filters out players who barely played (e.g., 2-8 minutes with artificially high RAPTOR)
df_merged["is_low_pick"] = (df_merged["draft_pick"] > 30) | (
    df_merged["was_drafted"] == 0
)
df_merged["is_high_nba_performer"] = (df_merged["war_total_sum"] >= 10.0) & (
    df_merged["mp_sum"] >= 2000
)
df_merged["is_diamond"] = df_merged["is_low_pick"] & df_merged["is_high_nba_performer"]

print(f"   ✅ Target: Drafted = {df_merged['target_drafted'].sum():,}")
print(
    f"   ✅ Target: Draft pick available = {df_merged['target_draft_pick'].notna().sum():,}"
)
print(f"   ✅ Low picks: {df_merged['is_low_pick'].sum():,}")
print(f"   ✅ High NBA performers: {df_merged['is_high_nba_performer'].sum():,}")
print(f"   ✅ Diamond players: {df_merged['is_diamond'].sum():,}")

# ============================================================================
# 9. Data Quality Summary
# ============================================================================
print("\n9️⃣ Data quality summary...")

print(f"\n   Missing values in key columns:")
key_cols = ["pts", "treb", "ast", "TS_per", "bpm", "year", "GP"]
for col in key_cols:
    if col in df_merged.columns:
        missing = df_merged[col].isna().sum()
        missing_pct = (missing / len(df_merged)) * 100
        print(f"   {col:15s}: {missing:6,} ({missing_pct:5.2f}%)")

# ============================================================================
# 10. Save Processed Data
# ============================================================================
print("\n🔟 Saving processed data...")

output_file = OUTPUT_DIR / "processed_data.csv"
df_merged.to_csv(output_file, index=False)
print(f"   ✅ Saved to: {output_file}")

# Save summary statistics
summary = {
    "total_players": len(df_merged),
    "drafted_players": df_merged["target_drafted"].sum(),
    "undrafted_players": (df_merged["target_drafted"] == 0).sum(),
    "players_with_nba_data": df_merged["raptor_total_mean"].notna().sum(),
    "diamond_players": df_merged["is_diamond"].sum(),
    "years_covered": f"{df_merged['year'].min()}-{df_merged['year'].max()}",
    "avg_years_in_college": f"{df_merged['years_in_college'].mean():.2f}",
    "features_count": len(df_merged.columns),
}

print("\n" + "=" * 80)
print("✅ DATA PROCESSING COMPLETE!")
print("=" * 80)
print("\n📊 Summary:")
for key, value in summary.items():
    print(f"   {key:25s}: {value}")

print(f"\n📁 Output file: {output_file}")
print(f"   Shape: {df_merged.shape}")
print(f"\n✅ Ready for feature engineering!")
