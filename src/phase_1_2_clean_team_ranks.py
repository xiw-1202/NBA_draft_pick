"""
Phase 1.2: Data Cleaning - Individual Files
Step 1.2.2: Restructure Final_Year_Team_Rank.csv
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
print("STEP 1.2.2: RESTRUCTURE TEAM RANKINGS DATA")
print("="*80)

# Sub-task a) Load file and inspect structure
print("\n📂 Loading Final_Year_Team_Rank.csv...")

# The file has a specific structure:
# Column 0: Rank for 2022
# Column 1: Team name for 2022
# Column 2: Empty
# Column 3: Rank for 2009
# Column 4: Team name for 2009

team_ranks_raw = pd.read_csv(RAW_DIR / 'Final_Year_Team_Rank.csv')
print(f"Raw shape: {team_ranks_raw.shape}")
print(f"Columns: {list(team_ranks_raw.columns)}")

print("\n🔍 First few rows of raw data:")
print(team_ranks_raw.head(10))

# Sub-task b) Identify which columns are years (2009-2022)
# Sub-task c) Melt dataframe from wide to long format
print("\n🔄 Restructuring data from wide to long format...")

all_team_ranks = []

# Process 2022 data
print("\nProcessing year 2022...")
# Skip the header row (row 0 which contains "Rk" and "Team")
team_2022 = team_ranks_raw.iloc[1:, [0, 1]].copy()  # Columns: Unnamed: 0 (rank), 2022 (team)
team_2022.columns = ['rank', 'team']
team_2022['year'] = 2022

# Clean and convert
team_2022['rank'] = pd.to_numeric(team_2022['rank'], errors='coerce')
team_2022 = team_2022.dropna(subset=['rank', 'team'])
team_2022 = team_2022[team_2022['team'] != 'Team']  # Remove any header rows

print(f"  Found {len(team_2022)} teams for 2022")
all_team_ranks.append(team_2022)

# Process 2009 data
print("\nProcessing year 2009...")
team_2009 = team_ranks_raw.iloc[1:, [3, 4]].copy()  # Columns: Unnamed: 3 (rank), 2009 (team)
team_2009.columns = ['rank', 'team']
team_2009['year'] = 2009

# Clean and convert
team_2009['rank'] = pd.to_numeric(team_2009['rank'], errors='coerce')
team_2009 = team_2009.dropna(subset=['rank', 'team'])

print(f"  Found {len(team_2009)} teams for 2009")
all_team_ranks.append(team_2009)

# Now let's check if there are more years in other files or if we need to handle this differently
# Based on the claude.md, it seems we need rankings for 2009-2021
# This file might only have endpoint years, let's work with what we have

# Combine all years
team_ranks_long = pd.concat(all_team_ranks, ignore_index=True)

# Sub-task d) Create columns: team, year, ap_rank, kenpom_rank
# Rename rank to ap_rank (assuming these are AP Poll rankings)
team_ranks_long.rename(columns={'rank': 'ap_rank'}, inplace=True)

# Add placeholder for KenPom rank
team_ranks_long['kenpom_rank'] = np.nan

# Sub-task e) Handle missing values (NaN = unranked)
# We've already removed NaN ranks, but let's ensure data types are correct
team_ranks_long['ap_rank'] = team_ranks_long['ap_rank'].astype(int)
team_ranks_long['year'] = team_ranks_long['year'].astype(int)

# Clean team names
team_ranks_long['team'] = team_ranks_long['team'].str.strip()

# Sort by year and rank
team_ranks_long = team_ranks_long.sort_values(['year', 'ap_rank']).reset_index(drop=True)

# Reorder columns
team_ranks_long = team_ranks_long[['team', 'year', 'ap_rank', 'kenpom_rank']]

print(f"\n✅ Restructured data shape: {team_ranks_long.shape}")

# Display summary
print("\n📊 Restructured Team Rankings Summary:")
print("-" * 40)
print(f"Total records: {len(team_ranks_long)}")
print(f"Years covered: {team_ranks_long['year'].unique()}")
print(f"Unique teams: {team_ranks_long['team'].nunique()}")
print(f"\nTeams per year:")
print(team_ranks_long['year'].value_counts().sort_index())

# Display sample
print("\n📋 Sample of restructured data (first 10 rows):")
print(team_ranks_long.head(10))

print("\n📋 Sample of restructured data (last 10 rows):")
print(team_ranks_long.tail(10))

# Check for data quality
print("\n✅ Data Quality Checks:")
print("-" * 40)
print(f"AP Rank range: {team_ranks_long['ap_rank'].min()} - {team_ranks_long['ap_rank'].max()}")

# Check for duplicate team-year combinations
duplicates = team_ranks_long[team_ranks_long.duplicated(subset=['team', 'year'], keep=False)]
if len(duplicates) > 0:
    print(f"\n⚠️ Found {len(duplicates)} duplicate team-year entries:")
    print(duplicates.head(10))
else:
    print("✅ No duplicate team-year combinations found")

# Get top teams by year
print("\n🏆 Top 5 teams by year:")
for year in team_ranks_long['year'].unique():
    print(f"\n{year}:")
    top_teams = team_ranks_long[team_ranks_long['year'] == year].head(5)
    for _, row in top_teams.iterrows():
        print(f"  {row['ap_rank']:2d}. {row['team']}")

# Note about limited years
print("\n⚠️ NOTE: This file only contains rankings for 2009 and 2022.")
print("   We may need additional data sources for years 2010-2021.")
print("   For now, we'll proceed with available data.")

# Sub-task f) Save as 'team_ranks_long.csv'
output_file = PROCESSED_DIR / 'team_ranks_long.csv'
team_ranks_long.to_csv(output_file, index=False)
print(f"\n💾 Restructured team rankings saved to: {output_file}")

# Also create a file to note missing years
missing_years = list(range(2010, 2022))
with open(PROCESSED_DIR / 'team_ranks_missing_years.txt', 'w') as f:
    f.write("Missing years in team rankings data:\n")
    f.write(str(missing_years))

print("\n" + "="*80)
print("✅ STEP 1.2.2 COMPLETE!")
print(f"Restructured data: {len(team_ranks_long)} team rankings")
print(f"⚠️ Note: Only 2009 and 2022 rankings available, years 2010-2021 missing")
print("="*80)
