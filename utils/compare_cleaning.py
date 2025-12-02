"""
Before/After Comparison - Show Cleaning Impact
==============================================
Compare original processed data vs cleaned data
"""

import pandas as pd
from pathlib import Path

# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "PROCESSED"

print("=" * 80)
print("BEFORE vs AFTER CLEANING COMPARISON")
print("=" * 80)

# Load both versions
print("\n📂 Loading data...")
df_before = pd.read_csv(DATA_DIR / "processed_data.csv", low_memory=False)
df_after = pd.read_csv(DATA_DIR / "processed_data_clean.csv", low_memory=False)

print(f"   Before: {df_before.shape}")
print(f"   After:  {df_after.shape}")

# ============================================================================
# Comparison Metrics
# ============================================================================

print("\n" + "=" * 80)
print("📊 COMPARISON METRICS")
print("=" * 80)

metrics = {
    "Total Rows": [len(df_before), len(df_after)],
    "Unique Players": [df_before["name_std"].nunique(), df_after["name_std"].nunique()],
    "Drafted Players": [df_before["was_drafted"].sum(), df_after["was_drafted"].sum()],
    "Diamond Players": [df_before["is_diamond"].sum(), df_after["is_diamond"].sum()],
    "Players with NBA Data": [
        df_before["raptor_total_mean"].notna().sum(),
        df_after["raptor_total_mean"].notna().sum(),
    ],
    "Duplicate Names": [
        len(df_before) - df_before["name_std"].nunique(),
        len(df_after) - df_after["name_std"].nunique(),
    ],
    "Total Missing Values": [
        df_before.isnull().sum().sum(),
        df_after.isnull().sum().sum(),
    ],
}

comparison = pd.DataFrame(metrics, index=["BEFORE", "AFTER"]).T
comparison["Change"] = comparison["AFTER"] - comparison["BEFORE"]

print("\n" + comparison.to_string())

# ============================================================================
# Name Standardization Examples
# ============================================================================

print("\n" + "=" * 80)
print("📝 NAME STANDARDIZATION EXAMPLES")
print("=" * 80)

# Find names that changed
print("\nChecking for duplicate name fixes...")

# Get duplicates from before
before_dupes = df_before["name_std"].value_counts()
before_dupes = before_dupes[before_dupes > 1]

if len(before_dupes) > 0:
    print(f"\n   Top duplicate names BEFORE cleaning:")
    for name in before_dupes.head(5).index:
        original_names = df_before[df_before["name_std"] == name][
            "player_name_clean"
        ].unique()
        print(f"   - '{name}' ({before_dupes[name]}x): {', '.join(original_names[:3])}")

        # Check if still duplicate after
        after_count = (df_after["name_std"] == name).sum()
        status = "✅ FIXED" if after_count <= 1 else "⚠️ STILL DUPLICATE"
        print(f"     AFTER: {after_count} entries {status}")

# ============================================================================
# Diamond Players Comparison
# ============================================================================

print("\n" + "=" * 80)
print("💎 DIAMOND PLAYERS COMPARISON")
print("=" * 80)

diamonds_before = df_before[df_before["is_diamond"] == True]
diamonds_after = df_after[df_after["is_diamond"] == True]

print(f"\nDiamond players BEFORE: {len(diamonds_before)}")
print(f"Diamond players AFTER:  {len(diamonds_after)}")
print(f"Change: {len(diamonds_after) - len(diamonds_before):+d}")

# Show top diamonds from cleaned data
print("\n   Top 10 Diamonds (AFTER cleaning):")
top_diamonds = diamonds_after.nlargest(10, "raptor_total_mean")
print(
    top_diamonds[
        ["player_name_clean", "year", "draft_pick", "raptor_total_mean"]
    ].to_string(index=False)
)

# ============================================================================
# Data Quality Improvement
# ============================================================================

print("\n" + "=" * 80)
print("✅ DATA QUALITY IMPROVEMENTS")
print("=" * 80)

print(f"\n   Rows removed (duplicates): {len(df_before) - len(df_after):,}")
print(
    f"   Missing values reduced: {df_before.isnull().sum().sum() - df_after.isnull().sum().sum():,}"
)
print(
    f"   Duplicate names eliminated: {(len(df_before) - df_before['name_std'].nunique()) - (len(df_after) - df_after['name_std'].nunique())}"
)

print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETE!")
print("=" * 80)
print(f"\n📁 Use cleaned data: {DATA_DIR / 'processed_data_clean.csv'}")
