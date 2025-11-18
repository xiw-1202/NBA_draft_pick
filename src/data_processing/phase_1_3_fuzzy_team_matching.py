"""
Phase 1.3: Key Creation for Merging
Step 1.3.2: Create fuzzy matching for team names
"""

import pandas as pd
import numpy as np
from pathlib import Path
from fuzzywuzzy import fuzz, process

# Set up paths
BASE_DIR = Path('/Users/sam/Documents/School/Emory/DataLab')
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

print("="*80)
print("STEP 1.3.2: CREATE FUZZY MATCHING FOR TEAM NAMES")
print("="*80)

# Load datasets
print("\n📂 Loading datasets...")
college_stats = pd.read_csv(PROCESSED_DIR / 'college_stats_with_keys.csv', low_memory=False)
draft_data = pd.read_csv(PROCESSED_DIR / 'draft_with_keys.csv')
team_ranks = pd.read_csv(PROCESSED_DIR / 'team_ranks_long.csv')

# Sub-task a) Extract unique team names from all datasets
print("\n🏫 Extracting unique team names from all datasets...")

college_teams = set(college_stats['team'].dropna().unique())
print(f"College stats: {len(college_teams)} unique teams")

draft_colleges = set(draft_data['college'].dropna().unique())
print(f"Draft data colleges: {len(draft_colleges)} unique colleges")

ranking_teams = set(team_ranks['team'].dropna().unique())
print(f"Team rankings: {len(ranking_teams)} unique teams")

# Combine all unique team names
all_teams = college_teams.union(draft_colleges).union(ranking_teams)
print(f"\nTotal unique team names across all datasets: {len(all_teams)}")

# Sub-task b) Create master team name list
print("\n📝 Creating master team name list...")

# Start with the most complete list (likely college teams)
# Sort alphabetically for easier review
master_teams = sorted(list(college_teams))

print(f"Master list initialized with {len(master_teams)} teams from college stats")

# Sub-task c) For each dataset's team names, find best match using fuzzywuzzy
print("\n🔍 Finding best matches for team names...")

# Function to clean team names for better matching
def clean_team_name(name):
    """Clean team name for better matching"""
    if pd.isna(name):
        return name
    
    # Convert to string and strip
    name = str(name).strip()
    
    # Common replacements
    replacements = {
        'St.': 'State',
        'State.': 'State',
        'U.': 'University',
        '&': 'and',
        '-': ' '
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    return name

# Create team mapping dictionary
team_mapping = {}

# Map draft college names to master list
print("\n🎯 Mapping draft college names...")
unmapped_draft = []

for college in draft_colleges:
    if college in master_teams:
        # Exact match
        team_mapping[college] = college
    else:
        # Fuzzy match
        match, score = process.extractOne(college, master_teams, scorer=fuzz.ratio)
        if score >= 80:  # 80% similarity threshold
            team_mapping[college] = match
            if score < 95:  # Show matches that aren't perfect
                print(f"  '{college}' → '{match}' (score: {score})")
        else:
            # Try with cleaned names
            clean_college = clean_team_name(college)
            clean_masters = [clean_team_name(t) for t in master_teams]
            match_idx = process.extractOne(clean_college, clean_masters, scorer=fuzz.ratio)
            if match_idx and match_idx[1] >= 75:
                team_mapping[college] = master_teams[clean_masters.index(match_idx[0])]
                print(f"  '{college}' → '{master_teams[clean_masters.index(match_idx[0])]}' (cleaned, score: {match_idx[1]})")
            else:
                unmapped_draft.append(college)
                print(f"  ⚠️ No good match for: '{college}'")

print(f"\nMapped {len(team_mapping)} draft colleges")
if unmapped_draft:
    print(f"Unmapped: {len(unmapped_draft)}")
    print("Sample unmapped draft colleges:")
    for team in unmapped_draft[:10]:
        print(f"  - {team}")

# Map ranking team names
print("\n🎯 Mapping ranking team names...")
unmapped_ranks = []

for team in ranking_teams:
    if team in master_teams:
        # Exact match
        team_mapping[team] = team
    elif team not in team_mapping:
        # Fuzzy match
        match, score = process.extractOne(team, master_teams, scorer=fuzz.ratio)
        if score >= 85:
            team_mapping[team] = match
            if score < 95:
                print(f"  '{team}' → '{match}' (score: {score})")
        else:
            unmapped_ranks.append(team)
            if score > 70:  # Show close misses
                print(f"  ⚠️ Weak match for '{team}' → '{match}' (score: {score})")

print(f"\nMapped teams from rankings: {len([t for t in ranking_teams if t in team_mapping])}")
if unmapped_ranks:
    print(f"Unmapped: {len(unmapped_ranks)}")

# Sub-task d) Create team_mapping dictionary
# Already created above, now save it

# Sub-task e) Apply mapping to standardize all team names
print("\n🔧 Applying team name standardization...")

# Create standardized team columns
college_stats['team_clean'] = college_stats['team'].map(lambda x: team_mapping.get(x, x))
draft_data['college_clean'] = draft_data['college'].map(lambda x: team_mapping.get(x, x))
team_ranks['team_clean'] = team_ranks['team'].map(lambda x: team_mapping.get(x, x))

# Sub-task f) Save mapping as 'team_name_mapping.csv'
print("\n💾 Saving team name mapping...")

# Convert mapping to DataFrame for saving
mapping_df = pd.DataFrame([
    {'original_name': k, 'standardized_name': v}
    for k, v in team_mapping.items()
])
mapping_df = mapping_df.sort_values('original_name').reset_index(drop=True)

mapping_file = PROCESSED_DIR / 'team_name_mapping.csv'
mapping_df.to_csv(mapping_file, index=False)
print(f"Team mapping saved to: {mapping_file}")

# Save datasets with cleaned team names
college_stats.to_csv(PROCESSED_DIR / 'college_stats_teams_cleaned.csv', index=False)
draft_data.to_csv(PROCESSED_DIR / 'draft_teams_cleaned.csv', index=False)
team_ranks.to_csv(PROCESSED_DIR / 'team_ranks_cleaned.csv', index=False)

# Analysis of standardization results
print("\n📊 Team Standardization Results:")
print("-" * 40)

# Check how many unique teams after standardization
print(f"Original unique teams in college: {college_stats['team'].nunique()}")
print(f"Cleaned unique teams in college: {college_stats['team_clean'].nunique()}")

print(f"\nOriginal unique colleges in draft: {draft_data['college'].nunique()}")
print(f"Cleaned unique colleges in draft: {draft_data['college_clean'].nunique()}")

print(f"\nOriginal unique teams in rankings: {team_ranks['team'].nunique()}")
print(f"Cleaned unique teams in rankings: {team_ranks['team_clean'].nunique()}")

# Show some interesting mappings
print("\n📋 Sample of interesting team mappings:")
interesting_mappings = mapping_df[mapping_df['original_name'] != mapping_df['standardized_name']].head(15)
for _, row in interesting_mappings.iterrows():
    print(f"  '{row['original_name']}' → '{row['standardized_name']}'")

print("\n" + "="*80)
print("✅ STEP 1.3.2 COMPLETE!")
print(f"Created team name mappings for {len(mapping_df)} team variations")
print("="*80)
