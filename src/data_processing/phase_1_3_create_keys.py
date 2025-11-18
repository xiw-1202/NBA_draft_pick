"""
Phase 1.3: Key Creation for Merging
Step 1.3.1: Create player-year keys
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path('/Users/sam/Documents/School/Emory/DataLab')
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

print("="*80)
print("STEP 1.3.1: CREATE PLAYER-YEAR KEYS")
print("="*80)

# Load cleaned datasets
print("\n📂 Loading cleaned datasets...")

# College stats with cleaned names
college_stats = pd.read_csv(PROCESSED_DIR / 'college_stats_names_cleaned.csv', low_memory=False)
print(f"College stats: {len(college_stats)} records")

# Draft data
draft_data = pd.read_csv(PROCESSED_DIR / 'draft_cleaned.csv')
print(f"Draft data: {len(draft_data)} records")

# Final year data
final_year_data = pd.read_csv(PROCESSED_DIR / 'final_year_names_cleaned.csv', low_memory=False)
print(f"Final year data: {len(final_year_data)} records")

# Sub-task a) In college_stats: create 'player_year_key' = player_name_clean + '_' + str(year)
print("\n🔑 Creating player-year keys in college stats...")

# Ensure year is integer
college_stats['year'] = pd.to_numeric(college_stats['year'], errors='coerce')
college_stats['player_year_key'] = (
    college_stats['player_name_clean'] + '_' + 
    college_stats['year'].astype('Int64').astype(str)
)

# Show sample keys
print("\n📋 Sample player-year keys from college stats:")
sample_keys = college_stats[['player_name_clean', 'year', 'player_year_key']].head(10)
print(sample_keys)

# Sub-task b) In draft_data: create same key format
print("\n🔑 Creating player-year keys in draft data...")

# First, we need to standardize the player names in draft data the same way
# Import the standardization function
import sys
sys.path.append(str(BASE_DIR / 'src'))

# Recreate the standardization function
import re

def standardize_player_name(name):
    """Standardize a player name"""
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    name = re.sub(r"[^a-zA-ZÀ-ÿ\s\-']", "", name)
    name = re.sub(r'\s+', ' ', name)
    name = name.title()
    name = re.sub(r"\bMc([a-z])", lambda m: f"Mc{m.group(1).upper()}", name)
    name = re.sub(r"\bO'([a-z])", lambda m: f"O'{m.group(1).upper()}", name)
    
    return name.strip()

# Standardize draft player names
draft_data['player_name_clean'] = draft_data['player_name'].apply(standardize_player_name)

# Create player-year key using draft year
draft_data['player_year_key'] = (
    draft_data['player_name_clean'] + '_' + 
    draft_data['draft_year'].astype('Int64').astype(str)
)

print("\n📋 Sample player-year keys from draft data:")
sample_draft_keys = draft_data[['player_name', 'player_name_clean', 'draft_year', 'player_year_key']].head(10)
print(sample_draft_keys)

# Sub-task c) Count unique keys in each dataset
print("\n📊 Unique keys analysis:")
print("-" * 40)

college_unique_keys = college_stats['player_year_key'].nunique()
draft_unique_keys = draft_data['player_year_key'].nunique()

print(f"College stats unique keys: {college_unique_keys}")
print(f"Draft data unique keys: {draft_unique_keys}")

# Check for duplicates in college stats
college_duplicates = college_stats[college_stats.duplicated(subset=['player_year_key'], keep=False)]
print(f"\nCollege stats duplicate keys: {len(college_duplicates)} records")

if len(college_duplicates) > 0:
    print("\n⚠️ Sample of duplicate keys in college stats:")
    dup_sample = college_duplicates.groupby('player_year_key').size().sort_values(ascending=False).head(10)
    for key, count in dup_sample.items():
        print(f"  {key}: {count} records")

# Sub-task d) Identify duplicate keys and handle them
print("\n🔍 Analyzing duplicate keys in detail...")

if len(college_duplicates) > 0:
    # Let's look at why there are duplicates
    sample_dup_key = college_duplicates['player_year_key'].iloc[0]
    sample_dup_records = college_stats[college_stats['player_year_key'] == sample_dup_key]
    
    print(f"\nExample duplicate: {sample_dup_key}")
    print(sample_dup_records[['player_name_clean', 'team', 'year', 'GP', 'pts']].to_string())
    
    # It's likely the same player on different teams in the same year (transfers)
    # We need to handle this by adding team to the key or selecting the record with most games played
    
    print("\n🔧 Handling duplicates by selecting record with most games played...")
    
    # For each duplicate key, keep the record with the most games played
    college_stats['GP'] = pd.to_numeric(college_stats['GP'], errors='coerce').fillna(0)
    
    # Sort by games played descending
    college_stats_sorted = college_stats.sort_values(['player_year_key', 'GP'], ascending=[True, False])
    
    # Keep first occurrence (most games) for each key
    college_stats_deduped = college_stats_sorted.drop_duplicates(subset=['player_year_key'], keep='first')
    
    print(f"After deduplication: {len(college_stats_deduped)} records (removed {len(college_stats) - len(college_stats_deduped)})")
    
    # Save the deduplicated version
    college_stats_deduped.to_csv(PROCESSED_DIR / 'college_stats_with_keys.csv', index=False)
else:
    college_stats_deduped = college_stats
    college_stats_deduped.to_csv(PROCESSED_DIR / 'college_stats_with_keys.csv', index=False)

# Also process final year data
print("\n🔑 Creating keys for final year data...")
final_year_data['year'] = pd.to_numeric(final_year_data['year'], errors='coerce')
final_year_data['player_year_key'] = (
    final_year_data['player_name_clean'] + '_' + 
    final_year_data['year'].astype('Int64').astype(str)
)

# Check for duplicates in final year data
final_duplicates = final_year_data[final_year_data.duplicated(subset=['player_year_key'], keep=False)]
print(f"Final year data duplicate keys: {len(final_duplicates)} records")

if len(final_duplicates) > 0:
    # Handle duplicates same way
    final_year_data['GP'] = pd.to_numeric(final_year_data['GP'], errors='coerce').fillna(0)
    final_year_sorted = final_year_data.sort_values(['player_year_key', 'GP'], ascending=[True, False])
    final_year_deduped = final_year_sorted.drop_duplicates(subset=['player_year_key'], keep='first')
    print(f"After deduplication: {len(final_year_deduped)} records (removed {len(final_year_data) - len(final_year_deduped)})")
else:
    final_year_deduped = final_year_data

final_year_deduped.to_csv(PROCESSED_DIR / 'final_year_with_keys.csv', index=False)

# Save draft data with keys
draft_data.to_csv(PROCESSED_DIR / 'draft_with_keys.csv', index=False)

# Quick check: How many draft players can we find in college stats?
print("\n🔍 Checking draft-college matching potential...")

draft_keys = set(draft_data['player_year_key'])
college_keys = set(college_stats_deduped['player_year_key'])
final_keys = set(final_year_deduped['player_year_key'])

matches_in_college = draft_keys.intersection(college_keys)
matches_in_final = draft_keys.intersection(final_keys)

print(f"\nDraft players found in college stats: {len(matches_in_college)}/{len(draft_keys)} ({len(matches_in_college)/len(draft_keys)*100:.1f}%)")
print(f"Draft players found in final year data: {len(matches_in_final)}/{len(draft_keys)} ({len(matches_in_final)/len(draft_keys)*100:.1f}%)")

# Show some unmatched draft players
unmatched = draft_keys - college_keys
if unmatched:
    print(f"\n📋 Sample of unmatched draft players (first 20):")
    unmatched_sample = list(unmatched)[:20]
    for key in unmatched_sample:
        draft_record = draft_data[draft_data['player_year_key'] == key].iloc[0]
        print(f"  {key} ({draft_record['college']})")

print("\n" + "="*80)
print("✅ STEP 1.3.1 COMPLETE!")
print(f"Keys created for all datasets:")
print(f"  - College stats: {len(college_stats_deduped)} records")
print(f"  - Final year: {len(final_year_deduped)} records")  
print(f"  - Draft data: {len(draft_data)} records")
print("="*80)
