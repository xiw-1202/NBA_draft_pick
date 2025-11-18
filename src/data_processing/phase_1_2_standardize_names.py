"""
Phase 1.2: Data Cleaning - Individual Files
Step 1.2.3: Standardize player names in college stats
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Set up paths
BASE_DIR = Path('/Users/sam/Documents/School/Emory/DataLab')
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

print("="*80)
print("STEP 1.2.3: STANDARDIZE PLAYER NAMES IN COLLEGE STATS")
print("="*80)

# Load college stats data
print("\n📂 Loading CollegeBasketballPlayers2009-2021.csv...")
college_stats = pd.read_csv(RAW_DIR / 'CollegeBasketballPlayers2009-2021.csv', low_memory=False)
print(f"Loaded {len(college_stats)} player-season records")

# Also load the final year data since it has the same structure
print("\n📂 Loading CollegePlayers_FinalYear_FULL.csv...")
final_year_data = pd.read_csv(RAW_DIR / 'CollegePlayers_FinalYear_FULL.csv', low_memory=False)
print(f"Loaded {len(final_year_data)} final year records")

def standardize_player_name(name):
    """
    Standardize a player name following the subtasks:
    a) Remove leading/trailing whitespaces
    b) Convert to title case
    c) Remove special characters (keep only letters, spaces, hyphens, apostrophes)
    d) Replace multiple spaces with single space
    """
    if pd.isna(name):
        return name
    
    # Convert to string if not already
    name = str(name)
    
    # Sub-task a) Remove leading/trailing whitespaces
    name = name.strip()
    
    # Sub-task c) Remove special characters (keep only letters, spaces, hyphens, apostrophes)
    # Keep: letters (including accented), spaces, hyphens, apostrophes
    name = re.sub(r"[^a-zA-ZÀ-ÿ\s\-']", "", name)
    
    # Sub-task d) Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Sub-task b) Convert to title case
    # Handle special cases like "O'Neal" or "McDonald"
    name = name.title()
    
    # Fix common title case issues
    # Fix Mc names (McDonald, McBride, etc.)
    name = re.sub(r"\bMc([a-z])", lambda m: f"Mc{m.group(1).upper()}", name)
    
    # Fix O' names
    name = re.sub(r"\bO'([a-z])", lambda m: f"O'{m.group(1).upper()}", name)
    
    return name.strip()

# Process college stats
print("\n🔧 Standardizing names in college stats...")

# Sub-task e) Create 'player_name_clean' column
college_stats['player_name_original'] = college_stats['player_name'].copy()
college_stats['player_name_clean'] = college_stats['player_name'].apply(standardize_player_name)

# Sub-task f) Log any names that changed significantly
print("\n📝 Analyzing name changes...")

# Find names that changed
name_changes = college_stats[['player_name_original', 'player_name_clean']].drop_duplicates()
changed_names = name_changes[name_changes['player_name_original'] != name_changes['player_name_clean']]

print(f"Total unique names: {len(name_changes)}")
print(f"Names that changed: {len(changed_names)}")

# Sample of significant changes
print("\n📋 Sample of name changes (first 20):")
for i, (_, row) in enumerate(changed_names.head(20).iterrows()):
    print(f"  '{row['player_name_original']}' → '{row['player_name_clean']}'")

# Look for problematic cases
print("\n🔍 Checking for potential issues...")

# Names with unusual characters
unusual_chars = college_stats[college_stats['player_name_original'].str.contains(r'[0-9@#$%^&*()_+=\[\]{}|\\/<>]', na=False)]
if len(unusual_chars) > 0:
    print(f"\n⚠️ Found {len(unusual_chars)} records with unusual characters in names:")
    print(unusual_chars[['player_name_original', 'player_name_clean']].head(10))

# Check for empty names after cleaning
empty_names = college_stats[college_stats['player_name_clean'] == '']
if len(empty_names) > 0:
    print(f"\n⚠️ Found {len(empty_names)} records with empty names after cleaning")

# Process final year data the same way
print("\n🔧 Standardizing names in final year data...")
final_year_data['player_name_original'] = final_year_data['player_name'].copy()
final_year_data['player_name_clean'] = final_year_data['player_name'].apply(standardize_player_name)

# Create a comprehensive name change log
print("\n📝 Creating name standardization log...")

all_name_changes = []

# From college stats
college_name_changes = college_stats[['player_name_original', 'player_name_clean']].drop_duplicates()
college_name_changes['source'] = 'college_stats'
all_name_changes.append(college_name_changes)

# From final year data
final_name_changes = final_year_data[['player_name_original', 'player_name_clean']].drop_duplicates()
final_name_changes['source'] = 'final_year'
all_name_changes.append(final_name_changes)

# Combine and deduplicate
name_change_log = pd.concat(all_name_changes, ignore_index=True)
name_change_log = name_change_log.drop_duplicates(subset=['player_name_original', 'player_name_clean'])

# Only keep records where name actually changed
name_change_log = name_change_log[name_change_log['player_name_original'] != name_change_log['player_name_clean']]
name_change_log = name_change_log.sort_values('player_name_original').reset_index(drop=True)

# Save the log
log_file = PROCESSED_DIR / 'player_name_standardization_log.csv'
name_change_log.to_csv(log_file, index=False)
print(f"💾 Name change log saved to: {log_file}")

# Save the cleaned datasets
print("\n💾 Saving cleaned datasets...")

# For college stats, keep the original column but use clean version
college_stats_output = PROCESSED_DIR / 'college_stats_names_cleaned.csv'
college_stats.to_csv(college_stats_output, index=False)
print(f"  College stats saved to: {college_stats_output}")

# For final year data
final_year_output = PROCESSED_DIR / 'final_year_names_cleaned.csv'
final_year_data.to_csv(final_year_output, index=False)
print(f"  Final year data saved to: {final_year_output}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n📊 College Stats:")
print(f"  Total records: {len(college_stats)}")
print(f"  Unique original names: {college_stats['player_name_original'].nunique()}")
print(f"  Unique clean names: {college_stats['player_name_clean'].nunique()}")
print(f"  Name reduction: {college_stats['player_name_original'].nunique() - college_stats['player_name_clean'].nunique()} names merged")

print(f"\n📊 Final Year Data:")
print(f"  Total records: {len(final_year_data)}")
print(f"  Unique original names: {final_year_data['player_name_original'].nunique()}")
print(f"  Unique clean names: {final_year_data['player_name_clean'].nunique()}")
print(f"  Name reduction: {final_year_data['player_name_original'].nunique() - final_year_data['player_name_clean'].nunique()} names merged")

print(f"\n📊 Name Change Log:")
print(f"  Total name changes logged: {len(name_change_log)}")

# Show some examples of standardization
print("\n📋 Examples of name standardization:")
examples = [
    "john smith",
    "JOHN SMITH",
    "John  Smith",
    "O'neal Johnson",
    "McDonald Williams",
    "José García"
]

for example in examples:
    standardized = standardize_player_name(example)
    print(f"  '{example}' → '{standardized}'")

print("\n✅ STEP 1.2.3 COMPLETE!")
print("="*80)
