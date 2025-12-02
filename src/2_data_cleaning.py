"""
Data Cleaning Patch - Fix Name Matching and Data Quality Issues
================================================================
Fixes:
1. Improved name standardization (handle spelling variants)
2. Remove duplicate players
3. Fix mixed dtypes
4. Better draft matching

Author: Sam
Date: December 2025
"""

import pandas as pd
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "PROCESSED"

print("=" * 80)
print("DATA CLEANING PATCH")
print("=" * 80)

# ============================================================================
# 1. Load Processed Data
# ============================================================================
print("\n1️⃣ Loading processed data...")

df = pd.read_csv(DATA_DIR / "processed_data.csv", low_memory=False)
print(f"   Loaded: {df.shape}")

# ============================================================================
# 2. Improved Name Standardization
# ============================================================================
print("\n2️⃣ Improving name standardization...")


def advanced_name_standardization(name):
    """
    Advanced name standardization to catch more variants
    Examples:
    - "JaQuori" vs "Jaquori" -> "jaquori"
    - "DeAndre" vs "De'Andre" -> "deandre"
    - "O'Brien" vs "OBrien" -> "obrien"
    """
    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    # Remove all apostrophes and hyphens in names
    name = name.replace("'", "").replace("-", "")

    # Remove suffixes (jr, sr, ii, iii, iv)
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", "", name)

    # Remove all punctuation except spaces
    name = re.sub(r"[^\w\s]", "", name)

    # Normalize multiple spaces to single space
    name = re.sub(r"\s+", " ", name).strip()

    # Remove common prefixes that might vary
    # (e.g., "De'Andre" vs "DeAndre" vs "Andre")
    # But keep the name intact for now

    return name


# Apply improved standardization
print("   Applying advanced name standardization...")
df["name_std_v2"] = df["player_name_clean"].apply(advanced_name_standardization)

# Check how many names changed
changed = (df["name_std"] != df["name_std_v2"]).sum()
print(f"   ✅ {changed:,} names improved")

# Replace old standardization
df["name_std"] = df["name_std_v2"]
df.drop(columns=["name_std_v2"], inplace=True)

# ============================================================================
# 3. Find and Merge Duplicate Players
# ============================================================================
print("\n3️⃣ Finding and merging duplicate players...")

# Find duplicates by standardized name
name_counts = df["name_std"].value_counts()
duplicates = name_counts[name_counts > 1]

print(f"   Found {len(duplicates)} players with multiple entries:")
if len(duplicates) > 0:
    print(f"   Top duplicates:")
    for name, count in duplicates.head(10).items():
        original_names = df[df["name_std"] == name]["player_name_clean"].unique()
        print(f"   - {name} ({count}x): {', '.join(original_names)}")

# Strategy: Keep the entry with the latest year (most recent/complete data)
print("\n   Merging duplicates (keeping latest year entry)...")

df_deduped = df.sort_values(["name_std", "year"], ascending=[True, False])
df_deduped = df_deduped.drop_duplicates(subset="name_std", keep="first")

print(f"   Before: {len(df):,} rows")
print(f"   After: {len(df_deduped):,} rows")
print(f"   ✅ Removed {len(df) - len(df_deduped):,} duplicate entries")

df = df_deduped.copy()

# ============================================================================
# 4. Fix Mixed Data Types
# ============================================================================
print("\n4️⃣ Fixing mixed data types...")

# Fix 'num' column (jersey number) - convert to numeric
if "num" in df.columns:
    # Convert to numeric, non-numeric becomes NaN
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    print(f"   ✅ Fixed 'num' column (jersey number)")

# Check for other mixed type columns
mixed_type_cols = []
for col in df.columns:
    if df[col].dtype == "object":
        # Try to convert to numeric
        numeric_version = pd.to_numeric(df[col], errors="coerce")
        non_numeric_count = numeric_version.isna().sum() - df[col].isna().sum()

        # If very few non-numeric values, it's likely a mixed type issue
        if non_numeric_count < len(df) * 0.1:  # Less than 10% non-numeric
            if col not in [
                "player_name",
                "player_name_clean",
                "name_std",
                "team",
                "conf",
                "yr",
                "ht",
                "position",
            ]:
                mixed_type_cols.append(col)

if mixed_type_cols:
    print(f"   Found {len(mixed_type_cols)} columns with mixed types:")
    for col in mixed_type_cols:
        print(f"   - {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"   ✅ Fixed {len(mixed_type_cols)} columns")

# ============================================================================
# 5. Re-validate Draft Matching
# ============================================================================
print("\n5️⃣ Re-validating draft matching...")

# Check draft pick consistency
drafted_players = df[df["was_drafted"] == 1]
print(f"   Drafted players: {len(drafted_players):,}")

# Check for players with draft_pick but was_drafted = 0
inconsistent = df[(df["draft_pick"].notna()) & (df["was_drafted"] == 0)]
if len(inconsistent) > 0:
    print(f"   ⚠️  Found {len(inconsistent)} inconsistent draft records")
    df.loc[inconsistent.index, "was_drafted"] = 1
    print(f"   ✅ Fixed inconsistent records")

# Check for players with was_drafted = 1 but no draft_pick
missing_pick = df[(df["was_drafted"] == 1) & (df["draft_pick"].isna())]
if len(missing_pick) > 0:
    print(f"   ⚠️  Found {len(missing_pick)} drafted players missing pick number")
    print(f"   Setting was_drafted = 0 for these players")
    df.loc[missing_pick.index, "was_drafted"] = 0

print(f"   ✅ Final drafted count: {df['was_drafted'].sum():,}")

# ============================================================================
# 6. Recalculate Diamond Players
# ============================================================================
print("\n6️⃣ Recalculating diamond players...")

# Recalculate with cleaned data
df["is_low_pick"] = (df["draft_pick"] > 30) | (df["was_drafted"] == 0)
df["is_high_nba_performer"] = df["raptor_total_mean"] > 0
df["is_diamond"] = df["is_low_pick"] & df["is_high_nba_performer"]

diamond_count = df["is_diamond"].sum()
print(f"   ✅ Diamond players: {diamond_count:,}")

# Show top diamonds
if diamond_count > 0:
    print("\n   Top 10 Diamond Players (by RAPTOR):")
    diamonds = df[df["is_diamond"] == True].sort_values(
        "raptor_total_mean", ascending=False
    )
    top_diamonds = diamonds.head(10)[
        ["player_name_clean", "year", "draft_pick", "pts", "bpm", "raptor_total_mean"]
    ]
    print(top_diamonds.to_string(index=False))

# ============================================================================
# 7. Clean Missing Values
# ============================================================================
print("\n7️⃣ Handling missing values...")

# Report missing values in key columns
key_cols = ["pts", "treb", "ast", "TS_per", "bpm", "GP", "Min_per"]
missing_report = []

for col in key_cols:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_report.append(
            {"column": col, "missing_count": missing_count, "missing_pct": missing_pct}
        )

missing_df = pd.DataFrame(missing_report)
print("\n   Missing values in key columns:")
print(missing_df.to_string(index=False))

# Fill missing values strategically
# For basic stats (pts, treb, ast) - fill with 0 (likely didn't play)
basic_stats = ["pts", "treb", "ast", "oreb", "dreb", "stl", "blk"]
for col in basic_stats:
    if col in df.columns:
        df[col].fillna(0, inplace=True)

# For percentages/rates - fill with median
percentage_cols = ["TS_per", "eFG", "FT_per", "twoP_per", "TP_per"]
for col in percentage_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# For BPM - fill with -5 (below average, but not extreme)
if "bpm" in df.columns:
    df["bpm"].fillna(-5.0, inplace=True)

print("   ✅ Filled missing values")

# ============================================================================
# 8. Data Quality Summary
# ============================================================================
print("\n8️⃣ Final data quality summary...")

summary = {
    "total_players": len(df),
    "unique_names": df["name_std"].nunique(),
    "drafted_players": df["was_drafted"].sum(),
    "players_with_nba_data": df["raptor_total_mean"].notna().sum(),
    "diamond_players": df["is_diamond"].sum(),
    "years_covered": f"{df['year'].min()}-{df['year'].max()}",
    "duplicate_names": len(df) - df["name_std"].nunique(),
    "total_missing_values": df.isnull().sum().sum(),
}

print("\n   📊 Data Quality Metrics:")
for key, value in summary.items():
    print(f"   {key:25s}: {value}")

# ============================================================================
# 9. Save Cleaned Data
# ============================================================================
print("\n9️⃣ Saving cleaned data...")

# Save cleaned data (overwrite processed_data.csv)
output_file = DATA_DIR / "processed_data_clean.csv"
df.to_csv(output_file, index=False)
print(f"   ✅ Saved to: {output_file}")

# Also save as new version for backup
backup_file = DATA_DIR / "processed_data_v1_cleaned.csv"
df.to_csv(backup_file, index=False)
print(f"   ✅ Backup saved to: {backup_file}")

# Save updated diamond players
diamonds = df[df["is_diamond"] == True]
diamonds.to_csv(DATA_DIR / "diamond_players_clean.csv", index=False)
print(f"   ✅ Saved {len(diamonds)} diamond players to: diamond_players_clean.csv")

# Create a changelog
changelog = f"""
DATA CLEANING PATCH - CHANGELOG
================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

CHANGES MADE:
1. Improved name standardization
   - Removed apostrophes and hyphens (De'Andre -> deandre)
   - Better handling of name variants
   - {changed:,} names improved

2. Merged duplicate players
   - Found {len(duplicates)} players with multiple entries
   - Kept most recent year entry for each player
   - Removed {len(df_deduped) - len(df):,} duplicate rows

3. Fixed mixed data types
   - Fixed 'num' column (jersey number)
   - Converted {len(mixed_type_cols)} columns to numeric

4. Re-validated draft matching
   - Ensured consistency between draft_pick and was_drafted
   - Final drafted count: {df['was_drafted'].sum():,}

5. Recalculated diamond players
   - New count: {df['is_diamond'].sum():,}

6. Filled missing values
   - Basic stats (pts, treb, ast): filled with 0
   - Percentages: filled with median
   - BPM: filled with -5.0

FINAL STATISTICS:
- Total players: {len(df):,}
- Unique names: {df['name_std'].nunique():,}
- Drafted: {df['was_drafted'].sum():,}
- Diamond players: {df['is_diamond'].sum():,}

FILES CREATED:
- processed_data_clean.csv (main cleaned file)
- processed_data_v1_cleaned.csv (backup)
- diamond_players_clean.csv (updated diamonds)
"""

changelog_file = DATA_DIR / "CLEANING_CHANGELOG.txt"
with open(changelog_file, "w") as f:
    f.write(changelog)

print(f"   ✅ Changelog saved to: {changelog_file}")

# ============================================================================
# 10. Verification
# ============================================================================
print("\n🔟 Verification checks...")

# Check 1: No duplicate names
dup_check = df["name_std"].duplicated().sum()
print(f"   ✅ Duplicate names: {dup_check} (should be 0)")

# Check 2: Consistent draft data
draft_consistency = df[(df["draft_pick"].notna()) & (df["was_drafted"] == 0)].shape[0]
print(f"   ✅ Draft inconsistencies: {draft_consistency} (should be 0)")

# Check 3: Missing values in key columns
missing_in_keys = df[key_cols].isnull().sum().sum()
print(f"   ✅ Missing values in key columns: {missing_in_keys}")

print("\n" + "=" * 80)
print("✅ DATA CLEANING PATCH COMPLETE!")
print("=" * 80)

print("\n📁 Next steps:")
print("   1. Review: cat Data/PROCESSED/CLEANING_CHANGELOG.txt")
print("   2. Check diamonds: head Data/PROCESSED/diamond_players_clean.csv")
print("   3. Ready for feature engineering!")
print(f"\n📊 Use this file for modeling: {output_file}")
