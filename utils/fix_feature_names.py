"""
Feature Name Cleaner - Improved Version
========================================
Clean feature names to be fully compatible with LightGBM (no spaces!)
"""

import pandas as pd
import re
from pathlib import Path

# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "PROCESSED"

print("=" * 80)
print("FEATURE NAME CLEANER - IMPROVED")
print("=" * 80)

# Load data
print("\n1️⃣ Loading featured data...")
df = pd.read_csv(DATA_DIR / "featured_data.csv")
print(f"   Loaded: {df.shape}")

# Load feature names
features = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()
print(f"   Features: {len(features)}")


# Improved function to clean feature names
def clean_feature_name(name):
    """Remove ALL special characters and spaces that LightGBM doesn't like"""
    # Replace spaces with underscores first
    name = name.replace(" ", "_")
    # Replace special characters with underscores
    name = re.sub(r"[^\w]", "_", name)
    # Replace multiple underscores with single
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    # Make sure it doesn't start with a number
    if name and name[0].isdigit():
        name = "n_" + name
    return name


# Clean all column names
print("\n2️⃣ Cleaning all column names...")
column_mapping = {col: clean_feature_name(col) for col in df.columns}

# Check for duplicates after cleaning
cleaned_names = list(column_mapping.values())
name_counts = {}
for name in cleaned_names:
    name_counts[name] = name_counts.get(name, 0) + 1

duplicates = [name for name, count in name_counts.items() if count > 1]

if duplicates:
    print(f"   ⚠️  Found {len(duplicates)} duplicate names after cleaning:")
    for dup in duplicates:
        originals = [k for k, v in column_mapping.items() if v == dup]
        print(f"   - {dup}: from {originals}")

    # Add suffixes to duplicates
    print("\n   Fixing duplicates with suffixes...")
    seen = {}
    for orig, cleaned in list(column_mapping.items()):
        if cleaned in duplicates:
            if cleaned not in seen:
                seen[cleaned] = 0
            else:
                seen[cleaned] += 1
                column_mapping[orig] = f"{cleaned}_{seen[cleaned]}"
    print("   ✅ Duplicates fixed")

# Apply renaming
df_cleaned = df.rename(columns=column_mapping)

# Update feature list
features_cleaned = [column_mapping.get(f, f) for f in features]

print(f"   ✅ Renamed {len(column_mapping)} columns")

# Show some examples
print("\n3️⃣ Examples of cleaned names:")
examples = [
    ("pos_Combo G", column_mapping.get("pos_Combo G", "pos_Combo G")),
    ("pos_Pure PG", column_mapping.get("pos_Pure PG", "pos_Pure PG")),
    ("Rec Rank", column_mapping.get("Rec Rank", "Rec Rank")),
    ("ast/tov", column_mapping.get("ast/tov", "ast_tov")),
    ("Unnamed: 64", column_mapping.get("Unnamed: 64", "Unnamed_64")),
]
for orig, cleaned in examples:
    if orig in column_mapping:
        print(f"   {orig:30s} → {cleaned}")

# Verify no special characters or spaces remain
print("\n4️⃣ Verifying cleaned names...")
remaining_issues = []
for col in df_cleaned.columns:
    # Check for any non-alphanumeric characters except underscore
    if re.search(r"[^\w]", col):
        remaining_issues.append(col)

if remaining_issues:
    print(f"   ❌ Still found {len(remaining_issues)} issues:")
    for issue in remaining_issues[:10]:
        print(f"   - '{issue}'")
    print(f"\n   This shouldn't happen - check the clean_feature_name function!")
else:
    print(f"   ✅ All names are LightGBM-compatible!")
    print(f"   ✅ No special characters or spaces remaining!")

# Save cleaned data
print("\n5️⃣ Saving cleaned data...")

output_file = DATA_DIR / "featured_data.csv"
df_cleaned.to_csv(output_file, index=False)
print(f"   ✅ Saved to: {output_file}")

# Save cleaned feature names
features_df = pd.DataFrame({"feature": features_cleaned})
features_df.to_csv(DATA_DIR / "feature_names.csv", index=False)
print(f"   ✅ Saved feature names to: feature_names.csv")

# Create mapping file for reference
mapping_df = pd.DataFrame(
    [{"original": k, "cleaned": v} for k, v in column_mapping.items() if k != v]
)
if len(mapping_df) > 0:
    mapping_df.to_csv(DATA_DIR / "feature_name_mapping.csv", index=False)
    print(f"   ✅ Saved name mapping to: feature_name_mapping.csv")
    print(f"\n   Changed {len(mapping_df)} column names")

print("\n" + "=" * 80)
print("✅ FEATURE NAMES FULLY CLEANED!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"   Total columns: {len(df_cleaned.columns)}")
print(f"   Total features: {len(features_cleaned)}")
print(f"   Names changed: {len(mapping_df) if len(mapping_df) > 0 else 0}")
print(f"   Special chars: ✅ REMOVED")
print(f"   Spaces: ✅ REMOVED")
print(f"   LightGBM-ready: ✅ YES")

print(f"\n🎯 Now run: python model_training.py")
