"""
Phase 1.2: Data Cleaning - Individual Files
Step 1.2.1: Clean DraftedPlayers2009-2021.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path('/Users/sam/Documents/School/Emory/DataLab')
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

print("="*80)
print("STEP 1.2.1: CLEAN DRAFTED PLAYERS DATA")
print("="*80)

# Sub-task a) Load Excel file and check for header duplicates
print("\n📂 Loading DraftedPlayers2009-2021.xlsx...")
draft_data_raw = pd.read_excel(RAW_DIR / 'DraftedPlayers2009-2021.xlsx')
print(f"Raw shape: {draft_data_raw.shape}")

# Display first few rows to confirm header issue
print("\n🔍 Checking first 5 rows for header duplicates:")
print(draft_data_raw.head())

# Sub-task b) Remove first row if it contains duplicate headers
print("\n🧹 Cleaning data...")

# Check if first row contains header-like values
first_row = draft_data_raw.iloc[0]
print(f"\nFirst row values: {first_row.values}")

# The first row contains NaN and header duplicates, remove it
if pd.isna(first_row['PLAYER']) or first_row['OVERALL'] == 'PICK':
    print("✅ Duplicate header row detected, removing...")
    draft_data = draft_data_raw.iloc[1:].copy()
    draft_data.reset_index(drop=True, inplace=True)
else:
    draft_data = draft_data_raw.copy()

print(f"Cleaned shape: {draft_data.shape}")

# Sub-task c) Rename columns to standard format
print("\n📝 Standardizing column names...")
column_rename = {
    'PLAYER': 'player_name',
    'TEAM': 'nba_team',
    'AFFILIATION': 'college',
    'YEAR': 'draft_year',
    'ROUND': 'round',
    'ROUND.1': 'round_pick',
    'OVERALL': 'overall_pick'
}

draft_data.rename(columns=column_rename, inplace=True)
print(f"New column names: {list(draft_data.columns)}")

# Sub-task d) Convert OVERALL column to integer
print("\n🔢 Converting data types...")

# First, let's check the data types and values
print("\nChecking overall_pick values before conversion:")
print(f"Data type: {draft_data['overall_pick'].dtype}")
print(f"Sample values: {draft_data['overall_pick'].head(10).values}")

# Convert to numeric, handling any non-numeric values
draft_data['overall_pick'] = pd.to_numeric(draft_data['overall_pick'], errors='coerce')
draft_data['round'] = pd.to_numeric(draft_data['round'], errors='coerce')
draft_data['round_pick'] = pd.to_numeric(draft_data['round_pick'], errors='coerce')

# Sub-task e) Create 'draft_year' column from YEAR (already renamed)
# Convert draft_year to integer
draft_data['draft_year'] = pd.to_numeric(draft_data['draft_year'], errors='coerce')

# Convert to integer types where appropriate
draft_data['overall_pick'] = draft_data['overall_pick'].astype('Int64')  # Nullable integer
draft_data['round'] = draft_data['round'].astype('Int64')
draft_data['round_pick'] = draft_data['round_pick'].astype('Int64')
draft_data['draft_year'] = draft_data['draft_year'].astype('Int64')

print("\n✅ Data types converted successfully")

# Display cleaned data summary
print("\n📊 Cleaned Draft Data Summary:")
print("-" * 40)
print(f"Total drafted players: {len(draft_data)}")
print(f"Years covered: {draft_data['draft_year'].min()} - {draft_data['draft_year'].max()}")
print(f"Players per year:")
print(draft_data['draft_year'].value_counts().sort_index())

# Check for missing values
print("\n🔍 Missing values check:")
missing_counts = draft_data.isnull().sum()
for col, count in missing_counts.items():
    if count > 0:
        print(f"  {col}: {count} ({count/len(draft_data)*100:.1f}%)")

# Display sample of cleaned data
print("\n📋 Sample of cleaned data (first 5 rows):")
print(draft_data.head())

print("\n📋 Sample of cleaned data (last 5 rows):")
print(draft_data.tail())

# Data quality checks
print("\n✅ Data Quality Checks:")
print("-" * 40)

# Check draft pick ranges
print(f"Overall pick range: {draft_data['overall_pick'].min()} - {draft_data['overall_pick'].max()}")
print(f"Round range: {draft_data['round'].min()} - {draft_data['round'].max()}")

# Check for duplicates
duplicate_players = draft_data[draft_data.duplicated(subset=['player_name', 'draft_year'], keep=False)]
if len(duplicate_players) > 0:
    print(f"\n⚠️ Found {len(duplicate_players)} potential duplicate entries:")
    print(duplicate_players[['player_name', 'draft_year', 'overall_pick']].head(10))
else:
    print("✅ No duplicate player-year combinations found")

# Sub-task f) Save as 'draft_cleaned.csv'
output_file = PROCESSED_DIR / 'draft_cleaned.csv'
draft_data.to_csv(output_file, index=False)
print(f"\n💾 Cleaned draft data saved to: {output_file}")

# Create a summary report
summary = {
    'total_players': len(draft_data),
    'years': f"{draft_data['draft_year'].min()}-{draft_data['draft_year'].max()}",
    'missing_values': missing_counts.to_dict(),
    'players_per_round': draft_data['round'].value_counts().sort_index().to_dict(),
    'columns': list(draft_data.columns),
    'data_types': draft_data.dtypes.astype(str).to_dict()
}

print("\n" + "="*80)
print("✅ STEP 1.2.1 COMPLETE!")
print(f"Cleaned draft data: {len(draft_data)} players from {draft_data['draft_year'].nunique()} drafts")
print("="*80)
