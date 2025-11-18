"""
Phase 1.1: Initial Data Loading and Inspection
Step 1.1.3: Identify column name variations and create mappings
"""

import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path('/Users/sam/Documents/School/Emory/DataLab')
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'

print("="*80)
print("STEP 1.1.3: IDENTIFY COLUMN NAME VARIATIONS")
print("="*80)

# Load all datasets
print("\n📂 Loading all datasets...")
college_stats = pd.read_csv(RAW_DIR / 'CollegeBasketballPlayers2009-2021.csv', low_memory=False)
draft_data = pd.read_excel(RAW_DIR / 'DraftedPlayers2009-2021.xlsx')
team_ranks = pd.read_csv(RAW_DIR / 'Final_Year_Team_Rank.csv')
raptor_data = pd.read_csv(RAW_DIR / 'modern_RAPTOR_by_player.csv')
final_year_data = pd.read_csv(RAW_DIR / 'CollegePlayers_FinalYear_FULL.csv', low_memory=False)

# Collect all column names
all_columns = {
    'college_stats': list(college_stats.columns),
    'draft_data': list(draft_data.columns),
    'team_ranks': list(team_ranks.columns),
    'raptor_data': list(raptor_data.columns),
    'final_year_data': list(final_year_data.columns)
}

print("\n📋 Column names by dataset:")
print("-" * 40)
for dataset_name, cols in all_columns.items():
    print(f"\n{dataset_name}:")
    print(f"  Total columns: {len(cols)}")
    print(f"  Sample columns: {cols[:5]} ...")

# Create column mappings for standardization
column_mappings = {
    # Player identification
    'player_name': {
        'standard': 'player_name',
        'variations': ['player_name', 'PLAYER', 'Player', 'player', 'Name'],
        'found_in': []
    },
    
    # Year/Season
    'season': {
        'standard': 'season',
        'variations': ['year', 'Year', 'YEAR', 'Season', 'season'],
        'found_in': []
    },
    
    # Team information
    'team': {
        'standard': 'team',
        'variations': ['team', 'Team', 'TEAM'],
        'found_in': []
    },
    
    # College/School affiliation
    'school': {
        'standard': 'school',
        'variations': ['AFFILIATION', 'school', 'team', 'college'],
        'found_in': []
    },
    
    # NBA Team
    'nba_team': {
        'standard': 'nba_team',
        'variations': ['TEAM', 'team', 'nba_team'],
        'found_in': []
    },
    
    # Draft information
    'draft_pick': {
        'standard': 'draft_pick',
        'variations': ['pick', 'OVERALL', 'Overall', 'draft_position'],
        'found_in': []
    },
    
    # Draft round
    'draft_round': {
        'standard': 'draft_round',
        'variations': ['ROUND', 'Round', 'round'],
        'found_in': []
    },
    
    # Conference
    'conference': {
        'standard': 'conference',
        'variations': ['conf', 'Conference', 'CONF'],
        'found_in': []
    },
    
    # Height
    'height': {
        'standard': 'height',
        'variations': ['ht', 'Height', 'height'],
        'found_in': []
    },
    
    # Games played
    'games_played': {
        'standard': 'games_played',
        'variations': ['GP', 'gp', 'games_played'],
        'found_in': []
    },
    
    # Points
    'points': {
        'standard': 'points',
        'variations': ['pts', 'PTS', 'Points', 'points'],
        'found_in': []
    },
    
    # Minutes
    'minutes': {
        'standard': 'minutes',
        'variations': ['mp', 'MP', 'Min_per', 'minutes'],
        'found_in': []
    },
    
    # Assists
    'assists': {
        'standard': 'assists',
        'variations': ['ast', 'AST', 'assists', 'Assists'],
        'found_in': []
    },
    
    # Rebounds
    'rebounds': {
        'standard': 'rebounds',
        'variations': ['treb', 'REB', 'rebounds', 'total_rebounds'],
        'found_in': []
    },
}

# Check which variations exist in which datasets
print("\n🔍 Checking for column variations across datasets...")
print("-" * 40)

for concept, mapping in column_mappings.items():
    print(f"\n{concept}:")
    for dataset_name, cols in all_columns.items():
        found_cols = []
        for col in cols:
            if col in mapping['variations'] or col.lower() in [v.lower() for v in mapping['variations']]:
                found_cols.append(col)
                if dataset_name not in mapping['found_in']:
                    mapping['found_in'].append(dataset_name)
        
        if found_cols:
            print(f"  {dataset_name}: {found_cols}")

# Identify potentially important columns that don't have mappings yet
print("\n🔍 Identifying unmapped columns...")
print("-" * 40)

all_mapped_variations = set()
for mapping in column_mappings.values():
    all_mapped_variations.update([v.lower() for v in mapping['variations']])

unmapped_columns = {}
for dataset_name, cols in all_columns.items():
    unmapped = [col for col in cols if col.lower() not in all_mapped_variations and not col.startswith('Unnamed')]
    if unmapped:
        unmapped_columns[dataset_name] = unmapped

print("\nPotentially important unmapped columns:")
for dataset_name, cols in unmapped_columns.items():
    print(f"\n{dataset_name}:")
    # Show first 10 unmapped columns
    for col in cols[:10]:
        print(f"  - {col}")
    if len(cols) > 10:
        print(f"  ... and {len(cols) - 10} more")

# Add additional mappings for important basketball statistics
additional_mappings = {
    # Shooting percentages
    'field_goal_pct': {
        'standard': 'fg_pct',
        'variations': ['eFG', 'FG_per', 'fg_pct'],
        'found_in': []
    },
    'true_shooting_pct': {
        'standard': 'ts_pct',
        'variations': ['TS_per', 'ts_pct', 'true_shooting'],
        'found_in': []
    },
    'three_point_pct': {
        'standard': 'three_pct',
        'variations': ['TP_per', '3P_per', 'three_pct'],
        'found_in': []
    },
    'free_throw_pct': {
        'standard': 'ft_pct',
        'variations': ['FT_per', 'ft_pct'],
        'found_in': []
    },
    
    # Advanced metrics
    'usage_rate': {
        'standard': 'usage_rate',
        'variations': ['usg', 'usage', 'usage_rate'],
        'found_in': []
    },
    'offensive_rating': {
        'standard': 'offensive_rating',
        'variations': ['Ortg', 'ortg', 'offensive_rating'],
        'found_in': []
    },
    'defensive_rating': {
        'standard': 'defensive_rating',
        'variations': ['drtg', 'Drtg', 'defensive_rating'],
        'found_in': []
    },
    
    # RAPTOR metrics
    'raptor_offense': {
        'standard': 'raptor_offense',
        'variations': ['raptor_offense', 'raptor_box_offense', 'raptor_onoff_offense'],
        'found_in': []
    },
    'raptor_defense': {
        'standard': 'raptor_defense',
        'variations': ['raptor_defense', 'raptor_box_defense', 'raptor_onoff_defense'],
        'found_in': []
    },
    'raptor_total': {
        'standard': 'raptor_total',
        'variations': ['raptor_total', 'raptor_box_total', 'raptor_onoff_total'],
        'found_in': []
    },
    'war': {
        'standard': 'war',
        'variations': ['war_total', 'war_reg_season', 'war_playoffs', 'WAR'],
        'found_in': []
    },
}

# Merge additional mappings
column_mappings.update(additional_mappings)

# Check additional mappings
print("\n🔍 Checking additional statistical column mappings...")
print("-" * 40)

for concept, mapping in additional_mappings.items():
    for dataset_name, cols in all_columns.items():
        found_cols = []
        for col in cols:
            if col in mapping['variations']:
                found_cols.append(col)
                if dataset_name not in mapping['found_in']:
                    mapping['found_in'].append(dataset_name)
        
        if found_cols:
            print(f"{concept} in {dataset_name}: {found_cols}")

# Save column mappings to JSON
output_file = PROCESSED_DIR / 'column_mappings.json'
with open(output_file, 'w') as f:
    json.dump(column_mappings, f, indent=2)

print(f"\n✅ Column mappings saved to: {output_file}")

# Create a summary report
print("\n" + "="*80)
print("COLUMN MAPPING SUMMARY")
print("="*80)

print("\n📊 Key findings:")
print("-" * 40)
print(f"Total mapping concepts defined: {len(column_mappings)}")
print(f"Datasets analyzed: {len(all_columns)}")

# Count how many concepts are found in each dataset
dataset_concept_counts = {}
for dataset_name in all_columns.keys():
    count = sum(1 for mapping in column_mappings.values() if dataset_name in mapping['found_in'])
    dataset_concept_counts[dataset_name] = count

print("\nConcepts found per dataset:")
for dataset_name, count in dataset_concept_counts.items():
    print(f"  {dataset_name}: {count}/{len(column_mappings)} concepts")

print("\n🎯 Important observations:")
print("-" * 40)
print("1. College stats and final year data have identical column structures")
print("2. Draft data uses uppercase column names (PLAYER, TEAM, etc.)")
print("3. Team ranks file has unusual structure - needs restructuring")
print("4. RAPTOR data has specialized NBA performance metrics")
print("5. The 'pick' column exists in college stats (mostly empty) for merging")

print("\n✅ STEP 1.1.3 COMPLETE!")
print("="*80)
