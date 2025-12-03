"""
Feature Engineering - NBA Draft Prediction
==========================================
Create advanced features for modeling from cleaned college basketball data.

Author: Sam
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "PROCESSED"
OUTPUT_DIR = DATA_DIR

print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# ============================================================================
# 1. Load Cleaned Data
# ============================================================================
print("\n1️⃣ Loading cleaned data...")

df = pd.read_csv(DATA_DIR / "processed_data_clean.csv")
print(f"   Loaded: {df.shape}")
print(f"   Players: {len(df):,}")
print(f"   Drafted: {df['was_drafted'].sum():,}")

# ============================================================================
# 2. Basic Statistics (Per Game)
# ============================================================================
print("\n2️⃣ Creating per-game statistics...")

# Points, rebounds, assists per game
df["ppg"] = df["pts"] / df["GP"]
df["rpg"] = df["treb"] / df["GP"]
df["apg"] = df["ast"] / df["GP"]
df["spg"] = df["stl"] / df["GP"]
df["bpg"] = df["blk"] / df["GP"]

# Offensive rebounds per game
df["orpg"] = df["oreb"] / df["GP"]
df["drpg"] = df["dreb"] / df["GP"]

print(f"   ✅ Created per-game stats: ppg, rpg, apg, spg, bpg")

# ============================================================================
# 3. Advanced Efficiency Metrics
# ============================================================================
print("\n3️⃣ Creating advanced efficiency metrics...")

# Player Efficiency Rating (simplified)
# PER = (pts + reb + ast + stl + blk) / min
df["per"] = (df["pts"] + df["treb"] + df["ast"] + df["stl"] + df["blk"]) / (
    df["GP"] * df["Min_per"] + 0.01
)

# True shooting attempts
df["tsa"] = df["twoPA"] + df["TPA"] + (0.44 * df["FTA"])

# Usage rate (already have 'usg', but ensure it's clean)
df["usage_rate"] = df["usg"]

# Assist-to-turnover ratio
df["ast_to_ratio"] = np.where(
    df["TO_per"] > 0, df["AST_per"] / df["TO_per"], df["AST_per"]
)

# Offensive rating (already have 'Ortg')
df["offensive_rating"] = df["Ortg"]

print(f"   ✅ Created efficiency metrics: per, tsa, ast_to_ratio")

# ============================================================================
# 4. Shooting Metrics
# ============================================================================
print("\n4️⃣ Creating shooting metrics...")

# True shooting percentage
df["ts_pct"] = df["TS_per"]

# Effective field goal percentage
df["efg_pct"] = df["eFG"]

# Two-point shooting
df["two_pt_pct"] = df["twoP_per"]
df["two_pt_rate"] = df["twoPA"] / (df["twoPA"] + df["TPA"] + 0.01)

# Three-point shooting
df["three_pt_pct"] = df["TP_per"]
df["three_pt_rate"] = df["TPA"] / (df["twoPA"] + df["TPA"] + 0.01)
df["three_pt_made_per_game"] = df["TPM"] / df["GP"]

# Free throw
df["ft_pct"] = df["FT_per"]
df["ft_rate"] = df["ftr"]
df["ft_made_per_game"] = df["FTM"] / df["GP"]

# Rim, mid-range, dunk efficiency
df["rim_pct"] = df["rimmade/(rimmade+rimmiss)"]
df["mid_pct"] = df["midmade/(midmade+midmiss)"]
df["dunk_pct"] = df["dunksmade/(dunksmade+dunksmiss)"]

print(f"   ✅ Created shooting metrics: ts_pct, efg_pct, 3pt%, rim%, mid%, dunk%")

# ============================================================================
# 5. Rebounding Metrics
# ============================================================================
print("\n5️⃣ Creating rebounding metrics...")

# Rebounding rates (already have ORB_per, DRB_per)
df["orb_rate"] = df["ORB_per"]
df["drb_rate"] = df["DRB_per"]
df["total_reb_rate"] = df["orb_rate"] + df["drb_rate"]

# Rebounding dominance
df["reb_dominance"] = df["rpg"] / df["Min_per"].replace(0, 1)

print(f"   ✅ Created rebounding metrics: orb_rate, drb_rate, reb_dominance")

# ============================================================================
# 6. Defensive Metrics
# ============================================================================
print("\n6️⃣ Creating defensive metrics...")

# Defensive rates
df["stl_rate"] = df["stl_per"]
df["blk_rate"] = df["blk_per"]

# Defensive efficiency
df["defensive_rating"] = df["drtg"]
df["adj_defensive_rating"] = df["adrtg"]

# Defensive contributions per game
df["defensive_contrib"] = df["spg"] + df["bpg"]

# Defensive BPM
df["dbpm"] = df.get("dbpm", 0)
df["obpm"] = df.get("obpm", 0)

print(f"   ✅ Created defensive metrics: stl_rate, blk_rate, defensive_rating")

# ============================================================================
# 7. Physical Attributes
# ============================================================================
print("\n7️⃣ Processing physical attributes...")


# Height - convert to inches
# ============================================================================
# NOTE: Height data is corrupted in the raw CSV files
# ============================================================================
# The 'ht' column in the raw data has been corrupted during CSV export/import.
# Heights like "6-2" (6 feet 2 inches) were converted to dates like "2-Jun".
# This makes the height data unusable, as all parsing attempts fail and result
# in the same median value (76.0 inches) for all players.
#
# RECOMMENDATION: Exclude height_inches from model features or obtain clean height data.
# ============================================================================

# COMMENTED OUT - Height feature is not usable due to data corruption
# def height_to_inches(ht):
#     """Convert height from various formats to inches"""
#     if pd.isna(ht):
#         return np.nan
#     ht = str(ht).strip()
#     # Format: "6-7" or "6'7" or "6-7"
#     if "-" in ht or "'" in ht:
#         parts = ht.replace("'", "-").replace('"', "").split("-")
#         if len(parts) >= 2:
#             try:
#                 feet = int(parts[0])
#                 inches = int(parts[1])
#                 return feet * 12 + inches
#             except:
#                 return np.nan
#     return np.nan
#
# df["height_inches"] = df["ht"].apply(height_to_inches)
# # Fill missing heights with position-based median
# if "position" in df.columns:
#     position_median_height = df.groupby("position")["height_inches"].transform("median")
#     df["height_inches"] = df["height_inches"].fillna(position_median_height)
# # If still missing, use overall median
# df["height_inches"] = df["height_inches"].fillna(df["height_inches"].median())
# print(f"   ✅ Processed height (median: {df['height_inches'].median():.1f} inches)")

print("   ⚠️  Height feature excluded (data corrupted in raw files)")

# ============================================================================
# 8. Experience & Development Features
# ============================================================================
print("\n8️⃣ Creating experience and development features...")

# Class year numeric
year_mapping = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4, "0": 0}
df["class_year_numeric"] = df["yr"].map(year_mapping)
df["class_year_numeric"] = df["class_year_numeric"].fillna(2.0)  # Default to sophomore

# Binary indicators
df["is_freshman"] = (df["yr"] == "Fr").astype(int)
df["is_sophomore"] = (df["yr"] == "So").astype(int)
df["is_junior"] = (df["yr"] == "Jr").astype(int)
df["is_senior"] = (df["yr"] == "Sr").astype(int)

# Years in college (already have this from data processing)
df["years_played"] = df["years_in_college"]

# Development indicators (improvement features already exist from data processing)
# Ensure we have the key improvement features
improvement_features = [
    col for col in df.columns if "improvement" in col or "career_avg" in col
]
print(f"   Found {len(improvement_features)} improvement/career features")

print(f"   ✅ Created experience features: class_year, is_senior, years_played")

# ============================================================================
# 9. Conference Strength
# ============================================================================
print("\n9️⃣ Creating conference features...")

# Major conferences (Power 6 + strong mid-majors)
major_conferences = ["ACC", "B10", "B12", "BE", "P12", "SEC", "Amer", "MWC", "WCC"]
df["is_major_conf"] = df["conf"].isin(major_conferences).astype(int)

# Conference average stats (proxy for strength)
conf_stats = (
    df.groupby("conf")
    .agg({"adjoe": "mean", "adrtg": "mean", "bpm": "mean"})
    .reset_index()
)

conf_stats.columns = ["conf", "conf_avg_offense", "conf_avg_defense", "conf_avg_bpm"]

df = df.merge(conf_stats, on="conf", how="left")

# Fill missing conference stats with overall median
df["conf_avg_offense"] = df["conf_avg_offense"].fillna(df["adjoe"].median())
df["conf_avg_defense"] = df["conf_avg_defense"].fillna(df["adrtg"].median())
df["conf_avg_bpm"] = df["conf_avg_bpm"].fillna(df["bpm"].median())

print(f"   ✅ Created conference features: is_major_conf, conf_avg_offense/defense")

# ============================================================================
# 10. Position Features
# ============================================================================
print("\n🔟 Creating position features...")

# Get position column (might be 'position' or 'Unnamed: 64')
position_col = "position" if "position" in df.columns else "Unnamed: 64"

if position_col in df.columns:
    # Fill missing positions with most common
    df[position_col] = df[position_col].fillna("Wing G")

    # One-hot encode positions
    position_dummies = pd.get_dummies(df[position_col], prefix="pos", dtype=int)
    df = pd.concat([df, position_dummies], axis=1)

    print(f"   ✅ Created {len(position_dummies.columns)} position dummy variables")
else:
    print(f"   ⚠️  Position column not found")

# ============================================================================
# 11. Consistency & Playing Time Features
# ============================================================================
print("\n1️⃣1️⃣ Creating consistency features...")

# Games played percentage (relative to max possible ~35 games)
df["games_played_pct"] = df["GP"] / 35.0
df["games_played_pct"] = df["games_played_pct"].clip(0, 1)

# Starter indicator (>25 min per game)
df["is_starter"] = (df["Min_per"] >= 25).astype(int)

# Minutes per game
df["mpg"] = df["Min_per"]

# Total minutes played in season
df["total_minutes"] = df["GP"] * df["Min_per"]

print(f"   ✅ Created consistency features: games_played_pct, is_starter, mpg")

# ============================================================================
# 12. Composite Scoring Features
# ============================================================================
print("\n1️⃣2️⃣ Creating composite features...")

# Scoring versatility (can score from multiple areas)
df["scoring_versatility"] = (
    (df["two_pt_pct"] > 0.45).astype(int)
    + (df["three_pt_pct"] > 0.35).astype(int)
    + (df["ft_pct"] > 0.75).astype(int)
)

# Well-rounded player score
df["well_rounded_score"] = (
    df["ppg"] / 20.0  # Scoring
    + df["rpg"] / 10.0  # Rebounding
    + df["apg"] / 5.0  # Passing
    + df["spg"] / 2.0  # Steals
    + df["bpg"] / 2.0  # Blocks
) / 5.0

# Efficiency composite
df["efficiency_composite"] = (
    df["ts_pct"] / 100.0
    + df["ast_to_ratio"] / 3.0
    + (df["bpm"] + 10) / 20.0  # Normalize BPM
) / 3.0

print(f"   ✅ Created composite features: scoring_versatility, well_rounded_score")

# ============================================================================
# 13. Handle Missing Values in New Features
# ============================================================================
print("\n1️⃣3️⃣ Handling missing values in features...")

# Get all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Count missing before
missing_before = df[numeric_cols].isnull().sum().sum()

# Fill strategy:
# 1. Rate/percentage features -> fill with 0 (no activity)
# 2. Per-game stats -> fill with 0 (didn't play/score)
# 3. Improvement features -> fill with 0 (no improvement data available)

rate_cols = [
    col
    for col in numeric_cols
    if any(x in col.lower() for x in ["_pct", "_rate", "pct", "rate"])
]
per_game_cols = [col for col in numeric_cols if col.endswith("pg")]
improvement_cols = [col for col in numeric_cols if "improvement" in col.lower()]

for col in rate_cols + per_game_cols + improvement_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Fill remaining numeric columns with median
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

missing_after = df[numeric_cols].isnull().sum().sum()

print(f"   Missing values before: {missing_before:,}")
print(f"   Missing values after: {missing_after:,}")
print(f"   ✅ Reduced missing values by {missing_before - missing_after:,}")

# ============================================================================
# 14. Feature Selection - Remove Data Leakage
# ============================================================================
print("\n1️⃣4️⃣ Selecting features (removing data leakage)...")

# Columns to EXCLUDE from modeling (identifiers, targets, NBA data)
exclude_cols = [
    # Identifiers
    "player_name",
    "player_name_clean",
    "name_std",
    "team",
    "conf",
    "position",
    "Unnamed: 64",
    "yr",
    "ht",
    # Target variables (these are what we predict)
    "was_drafted",
    "target_drafted",
    "draft_pick",
    "target_draft_pick",
    "draft_year",
    "draft_round",
    "draft_team",
    "pick",
    "is_diamond",
    "is_low_pick",
    "is_high_nba_performer",
    # NBA data (future information, not available at draft time)
    "raptor_total_mean",
    "raptor_offense_mean",
    "raptor_defense_mean",
    "war_total_sum",
    "war_reg_season_sum",
    "season_count",
    "season_min",
    "season_max",
    "mp_sum",
    "poss_sum",
    # Redundant/intermediate columns
    "name_std_v2",
    "player_id",
    "pid",
    "type",
    "pts_prev",
    "treb_prev",
    "ast_prev",
    "TS_per_prev",
    "bpm_prev",
    "Min_per_prev",
    "usg_prev",
    "pts_first_year",
    "treb_first_year",
    "ast_first_year",
    "TS_per_first_year",
    "bpm_first_year",
    "Min_per_first_year",
    "usg_first_year",
]

# Get all columns
all_cols = df.columns.tolist()

# Feature columns = all columns - excluded columns
feature_cols = [col for col in all_cols if col not in exclude_cols]

# Further filter: only numeric columns (for modeling)
feature_cols = [col for col in feature_cols if df[col].dtype in ["int64", "float64"]]

print(f"   Total columns: {len(all_cols)}")
print(f"   Excluded columns: {len(exclude_cols)}")
print(f"   Feature columns: {len(feature_cols)}")

# ============================================================================
# 15. Create Final Dataset
# ============================================================================
print("\n1️⃣5️⃣ Creating final feature dataset...")

# Clean feature names for LightGBM compatibility (remove special JSON characters)
print("   Cleaning feature names...")


def clean_feature_name(name):
    """Remove special characters that cause issues with LightGBM JSON serialization"""
    name = name.replace("/", "_div_")
    name = name.replace("(", "_")
    name = name.replace(")", "_")
    name = name.replace("+", "_plus_")
    name = name.replace("-", "_minus_")
    name = name.replace("*", "_times_")
    name = name.replace(" ", "_")
    name = name.replace("[", "_")
    name = name.replace("]", "_")
    name = name.replace("{", "_")
    name = name.replace("}", "_")
    name = name.replace('"', "_")
    name = name.replace("'", "_")
    name = name.replace("\\", "_")
    name = name.replace(":", "_")
    # Remove consecutive underscores
    while "__" in name:
        name = name.replace("__", "_")
    # Remove trailing underscores
    name = name.strip("_")
    return name


# Create mapping of old to new names
name_mapping = {col: clean_feature_name(col) for col in df.columns}

# Apply name cleaning
df = df.rename(columns=name_mapping)

# Update feature_cols list with cleaned names
feature_cols = [clean_feature_name(col) for col in feature_cols]

print(f"   ✅ Cleaned {len([k for k, v in name_mapping.items() if k != v])} feature names")

# Essential columns to keep alongside features
essential_cols = [
    "player_name_clean",
    "year",
    "team",
    "conf",
    "yr",
    "position" if "position" in df.columns else "Unnamed_64",
    "was_drafted",
    "draft_pick",
    "is_diamond",
    "raptor_total_mean",
    "war_total_sum",  # Keep for diamond analysis
]

# Ensure essential columns exist (with cleaned names)
essential_cols = [clean_feature_name(col) for col in essential_cols if clean_feature_name(col) in df.columns]

# Final dataset = essential + features
final_cols = essential_cols + feature_cols

df_final = df[final_cols].copy()

print(f"   Final dataset shape: {df_final.shape}")
print(f"   Features for modeling: {len(feature_cols)}")

# ============================================================================
# 16. Save Feature Engineered Data
# ============================================================================
print("\n1️⃣6️⃣ Saving feature-engineered dataset...")

# Save full dataset
output_file = OUTPUT_DIR / "featured_data.csv"
df_final.to_csv(output_file, index=False)
print(f"   ✅ Saved to: {output_file}")

# Save feature list
feature_list_file = OUTPUT_DIR / "feature_list.txt"
with open(feature_list_file, "w") as f:
    f.write("FEATURE LIST FOR MODELING\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total Features: {len(feature_cols)}\n\n")

    # Categorize features
    categories = {
        "Per-Game Stats": [c for c in feature_cols if c.endswith("pg")],
        "Shooting Metrics": [
            c
            for c in feature_cols
            if any(x in c for x in ["_pct", "pt_", "ft_", "efg", "ts_"])
        ],
        "Efficiency": [
            c
            for c in feature_cols
            if any(x in c for x in ["efficiency", "rating", "per", "bpm"])
        ],
        "Rebounding": [
            c for c in feature_cols if any(x in c for x in ["reb", "orb", "drb"])
        ],
        "Defense": [
            c
            for c in feature_cols
            if any(x in c for x in ["stl", "blk", "defensive", "drtg"])
        ],
        "Experience": [
            c
            for c in feature_cols
            if any(x in c for x in ["year", "class", "senior", "freshman"])
        ],
        "Development": [
            c for c in feature_cols if "improvement" in c or "career_avg" in c
        ],
        "Conference": [c for c in feature_cols if "conf" in c],
        "Position": [c for c in feature_cols if c.startswith("pos_")],
        "Physical": [c for c in feature_cols if "height" in c],
        "Other": [],
    }

    # Assign uncategorized features
    categorized = set()
    for cat_features in categories.values():
        categorized.update(cat_features)

    categories["Other"] = [c for c in feature_cols if c not in categorized]

    # Write categorized features
    for category, features in categories.items():
        if features:
            f.write(f"\n{category} ({len(features)} features):\n")
            f.write("-" * 80 + "\n")
            for feat in sorted(features):
                f.write(f"  - {feat}\n")

print(f"   ✅ Feature list saved to: {feature_list_file}")

# Save feature names as CSV for easy loading
feature_names_df = pd.DataFrame({"feature": feature_cols})
feature_names_df.to_csv(OUTPUT_DIR / "feature_names.csv", index=False)
print(f"   ✅ Feature names saved to: feature_names.csv")

# ============================================================================
# 17. Summary Statistics
# ============================================================================
print("\n1️⃣7️⃣ Summary statistics...")

summary = {
    "total_players": len(df_final),
    "drafted_players": df_final["was_drafted"].sum(),
    "diamond_players": df_final["is_diamond"].sum(),
    "total_features": len(feature_cols),
    "per_game_features": len([c for c in feature_cols if c.endswith("pg")]),
    "shooting_features": len([c for c in feature_cols if "_pct" in c or "_rate" in c]),
    "improvement_features": len([c for c in feature_cols if "improvement" in c]),
    "position_features": len([c for c in feature_cols if c.startswith("pos_")]),
    "missing_values": df_final[feature_cols].isnull().sum().sum(),
}

print("\n   📊 Feature Engineering Summary:")
for key, value in summary.items():
    print(f"   {key:25s}: {value}")

# Show feature categories breakdown
print("\n   📋 Feature Categories:")
for category, features in categories.items():
    if features:
        print(f"   {category:20s}: {len(features):3d} features")

print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 80)

print(f"\n📁 Output files:")
print(f"   - {output_file}")
print(f"   - {feature_list_file}")
print(f"   - {OUTPUT_DIR / 'feature_names.csv'}")

print(f"\n🎯 Ready for modeling with {len(feature_cols)} features!")
print(f"   Use 'featured_data.csv' for training")
