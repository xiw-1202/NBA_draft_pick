"""
Feature Preview - Explore Created Features
==========================================
Quick exploration of feature-engineered dataset
"""

import pandas as pd
from pathlib import Path

# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "PROCESSED"

print("=" * 80)
print("FEATURE PREVIEW")
print("=" * 80)

# Load featured data
df = pd.read_csv(DATA_DIR / "featured_data.csv")
print(f"\n📊 Dataset: {df.shape}")

# Load feature names
features = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()
print(f"📋 Features for modeling: {len(features)}")

# ============================================================================
# 1. Show Sample Players
# ============================================================================
print("\n" + "=" * 80)
print("1️⃣ SAMPLE PLAYERS (with key features)")
print("=" * 80)

sample_cols = [
    "player_name_clean",
    "year",
    "was_drafted",
    "draft_pick",
    "ppg",
    "rpg",
    "apg",
    "ts_pct",
    "bpm",
    "height_inches",
    "is_senior",
    "is_major_conf",
]

print("\n   Top 5 Drafted Players (by BPM):")
drafted = df[df["was_drafted"] == 1].sort_values("bpm", ascending=False).head(5)
print(drafted[sample_cols].to_string(index=False))

print("\n   Top 5 Undrafted Players (by BPM):")
undrafted = df[df["was_drafted"] == 0].sort_values("bpm", ascending=False).head(5)
print(undrafted[sample_cols].to_string(index=False))

# ============================================================================
# 2. Feature Statistics
# ============================================================================
print("\n" + "=" * 80)
print("2️⃣ FEATURE STATISTICS")
print("=" * 80)

# Key features statistics
key_features = [
    "ppg",
    "rpg",
    "apg",
    "ts_pct",
    "bpm",
    "per",
    "three_pt_pct",
    "usage_rate",
    "height_inches",
]

stats_df = df[key_features].describe().T
stats_df["missing"] = df[key_features].isnull().sum()

print("\n   Key Feature Statistics:")
print(stats_df[["mean", "50%", "std", "min", "max", "missing"]].round(2).to_string())

# ============================================================================
# 3. Improvement Features Check
# ============================================================================
print("\n" + "=" * 80)
print("3️⃣ IMPROVEMENT FEATURES (Late Bloomers)")
print("=" * 80)

improvement_cols = [col for col in df.columns if "total_improvement" in col]

if improvement_cols:
    print(f"\n   Found {len(improvement_cols)} improvement features")

    # Show players with biggest improvements
    if "pts_total_improvement" in df.columns:
        print("\n   Players with Biggest PPG Improvement:")
        top_improvers = df.nlargest(5, "pts_total_improvement")
        improve_cols = [
            "player_name_clean",
            "year",
            "pts_total_improvement",
            "bpm_total_improvement",
            "was_drafted",
            "draft_pick",
        ]
        print(top_improvers[improve_cols].to_string(index=False))

# ============================================================================
# 4. Position Distribution
# ============================================================================
print("\n" + "=" * 80)
print("4️⃣ POSITION DISTRIBUTION")
print("=" * 80)

position_cols = [col for col in features if col.startswith("pos_")]

if position_cols:
    print(f"\n   Position dummy variables: {len(position_cols)}")

    # Count players by position
    for pos_col in sorted(position_cols):
        count = df[pos_col].sum()
        drafted_pct = df[df[pos_col] == 1]["was_drafted"].mean() * 100
        print(f"   {pos_col:20s}: {count:5,} players ({drafted_pct:4.1f}% drafted)")

# ============================================================================
# 5. Conference Analysis
# ============================================================================
print("\n" + "=" * 80)
print("5️⃣ CONFERENCE STRENGTH")
print("=" * 80)

major_conf = df[df["is_major_conf"] == 1]
other_conf = df[df["is_major_conf"] == 0]

print(
    f"\n   Major conferences: {len(major_conf):,} players ({len(major_conf)/len(df)*100:.1f}%)"
)
print(
    f"   Other conferences: {len(other_conf):,} players ({len(other_conf)/len(df)*100:.1f}%)"
)

print(f"\n   Draft rate by conference:")
print(f"   Major conferences: {major_conf['was_drafted'].mean()*100:.2f}%")
print(f"   Other conferences: {other_conf['was_drafted'].mean()*100:.2f}%")

# ============================================================================
# 6. Class Year Analysis
# ============================================================================
print("\n" + "=" * 80)
print("6️⃣ CLASS YEAR ANALYSIS")
print("=" * 80)

class_features = ["is_freshman", "is_sophomore", "is_junior", "is_senior"]
class_names = ["Freshman", "Sophomore", "Junior", "Senior"]

print(f"\n   Draft rate by class year:")
for feat, name in zip(class_features, class_names):
    if feat in df.columns:
        class_df = df[df[feat] == 1]
        draft_rate = class_df["was_drafted"].mean() * 100
        avg_ppg = class_df["ppg"].mean()
        avg_bpm = class_df["bpm"].mean()
        print(
            f"   {name:15s}: {len(class_df):5,} players | {draft_rate:5.2f}% drafted | {avg_ppg:5.1f} PPG | {avg_bpm:5.1f} BPM"
        )

# ============================================================================
# 7. Diamond Players Preview
# ============================================================================
print("\n" + "=" * 80)
print("7️⃣ DIAMOND PLAYERS")
print("=" * 80)

diamonds = df[df["is_diamond"] == True]

if len(diamonds) > 0:
    print(f"\n   Total diamond players: {len(diamonds)}")
    print(f"\n   Top 10 Diamonds (by RAPTOR):")

    diamond_cols = [
        "player_name_clean",
        "year",
        "draft_pick",
        "ppg",
        "bpm",
        "ts_pct",
        "raptor_total_mean",
    ]
    print(
        diamonds.nlargest(10, "raptor_total_mean")[diamond_cols].to_string(index=False)
    )

# ============================================================================
# 8. Missing Values Report
# ============================================================================
print("\n" + "=" * 80)
print("8️⃣ MISSING VALUES IN FEATURES")
print("=" * 80)

missing_counts = df[features].isnull().sum()
features_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)

if len(features_with_missing) > 0:
    print(f"\n   ⚠️  {len(features_with_missing)} features with missing values:")
    print(features_with_missing.head(10))
else:
    print(f"\n   ✅ No missing values in feature columns!")

# ============================================================================
# 9. Ready for Modeling
# ============================================================================
print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING SUMMARY")
print("=" * 80)

print(
    f"""
   Total Players:          {len(df):,}
   Drafted:               {df['was_drafted'].sum():,} ({df['was_drafted'].mean()*100:.2f}%)
   Diamond Players:       {df['is_diamond'].sum():,}
   
   Features:              {len(features)}
   Missing Values:        {df[features].isnull().sum().sum():,}
   
   Ready for:
   ✅ Time-based split (train 2009-2018, test 2019-2021)
   ✅ LightGBM Classifier (predict drafted vs undrafted)
   ✅ LightGBM Regressor (predict draft pick 1-60)
   ✅ Diamond detection
"""
)

print("=" * 80)
print("🚀 READY FOR MODELING!")
print("=" * 80)
