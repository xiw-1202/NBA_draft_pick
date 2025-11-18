"""
Phase 1.3: Key Creation for Merging
Step 1.3.3: Identify final college season for each player
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path('/Users/sam/Documents/School/Emory/DataLab')
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

print("="*80)
print("STEP 1.3.3: IDENTIFY FINAL COLLEGE SEASON FOR EACH PLAYER")
print("="*80)

# Load college stats with cleaned names and teams
print("\n📂 Loading college stats data...")
college_stats = pd.read_csv(PROCESSED_DIR / 'college_stats_teams_cleaned.csv', low_memory=False)
print(f"Loaded {len(college_stats)} player-season records")

# Ensure year is numeric
college_stats['year'] = pd.to_numeric(college_stats['year'], errors='coerce')

# Sub-task a) Group college_stats by player_name_clean
print("\n👥 Grouping players to find career spans...")

player_careers = college_stats.groupby('player_name_clean').agg({
    'year': ['min', 'max', 'count'],
    'GP': 'sum',
    'pts': 'mean'
}).round(2)

player_careers.columns = ['first_year', 'final_year', 'seasons_played', 'total_games', 'avg_pts']
player_careers = player_careers.reset_index()

print(f"Found {len(player_careers)} unique players")

# Sub-task b) For each player, find the maximum year
# Already done in the aggregation above as 'final_year'

# Sub-task c) Create boolean column 'is_final_season'
print("\n🎓 Marking final seasons...")

# Create a set of player-year combinations for final seasons
final_season_keys = set()
for _, player in player_careers.iterrows():
    key = f"{player['player_name_clean']}_{int(player['final_year'])}"
    final_season_keys.add(key)

# Mark final seasons in college_stats
college_stats['is_final_season'] = college_stats['player_year_key'].isin(final_season_keys)

print(f"Marked {college_stats['is_final_season'].sum()} records as final seasons")
print(f"Percentage of records that are final seasons: {college_stats['is_final_season'].mean()*100:.1f}%")

# Sub-task d) Count players with multiple seasons
print("\n📊 Analyzing player careers...")

multi_season_players = player_careers[player_careers['seasons_played'] > 1]
print(f"Players with multiple seasons: {len(multi_season_players)} ({len(multi_season_players)/len(player_careers)*100:.1f}%)")

# Distribution of seasons played
seasons_dist = player_careers['seasons_played'].value_counts().sort_index()
print("\nDistribution of seasons played:")
for seasons, count in seasons_dist.items():
    pct = count/len(player_careers)*100
    print(f"  {int(seasons)} season(s): {count:5d} players ({pct:5.1f}%)")

# Players with most seasons
print("\n🏆 Players with most seasons:")
longest_careers = player_careers.nlargest(10, 'seasons_played')[['player_name_clean', 'seasons_played', 'first_year', 'final_year']]
for _, player in longest_careers.iterrows():
    print(f"  {player['player_name_clean']}: {int(player['seasons_played'])} seasons ({int(player['first_year'])}-{int(player['final_year'])})")

# Sub-task e) Save player career summary as 'player_careers.csv'
print("\n💾 Saving player career summary...")

# Add some additional useful stats
player_careers['career_span'] = player_careers['final_year'] - player_careers['first_year'] + 1
player_careers['is_one_and_done'] = player_careers['seasons_played'] == 1

# Sort by number of seasons and points for interest
player_careers = player_careers.sort_values(['seasons_played', 'avg_pts'], ascending=[False, False])

career_file = PROCESSED_DIR / 'player_careers.csv'
player_careers.to_csv(career_file, index=False)
print(f"Player careers saved to: {career_file}")

# Also save the college stats with final season flag
college_stats_file = PROCESSED_DIR / 'college_stats_with_final_flag.csv'
college_stats.to_csv(college_stats_file, index=False)
print(f"College stats with final season flag saved to: {college_stats_file}")

# Create a dataset of just final seasons
print("\n📝 Creating final seasons dataset...")
final_seasons_only = college_stats[college_stats['is_final_season']].copy()
print(f"Final seasons dataset: {len(final_seasons_only)} records")

# Compare with the provided final year file
final_year_provided = pd.read_csv(PROCESSED_DIR / 'final_year_with_keys.csv', low_memory=False)
print(f"Provided final year file: {len(final_year_provided)} records")

# Check overlap
our_final_keys = set(final_seasons_only['player_year_key'])
provided_final_keys = set(final_year_provided['player_year_key'])

overlap = our_final_keys.intersection(provided_final_keys)
only_ours = our_final_keys - provided_final_keys
only_provided = provided_final_keys - our_final_keys

print(f"\n🔍 Comparison with provided final year file:")
print(f"  Common players: {len(overlap)}")
print(f"  Only in our detection: {len(only_ours)}")
print(f"  Only in provided file: {len(only_provided)}")

# Save our version of final seasons
final_seasons_file = PROCESSED_DIR / 'final_seasons_detected.csv'
final_seasons_only.to_csv(final_seasons_file, index=False)
print(f"\n💾 Final seasons dataset saved to: {final_seasons_file}")

# Analysis summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
📊 Career Analysis Results:
  - Total unique players: {len(player_careers):,}
  - Players with 1 season: {(player_careers['seasons_played'] == 1).sum():,} ({(player_careers['seasons_played'] == 1).mean()*100:.1f}%)
  - Players with 2+ seasons: {(player_careers['seasons_played'] >= 2).sum():,} ({(player_careers['seasons_played'] >= 2).mean()*100:.1f}%)
  - Players with 4+ seasons: {(player_careers['seasons_played'] >= 4).sum():,} ({(player_careers['seasons_played'] >= 4).mean()*100:.1f}%)
  
  - Average seasons per player: {player_careers['seasons_played'].mean():.2f}
  - Maximum seasons played: {int(player_careers['seasons_played'].max())}
  
  - Final season records identified: {len(final_seasons_only):,}
  - Match with provided final year file: {len(overlap)/len(provided_final_keys)*100:.1f}%
""")

print("✅ STEP 1.3.3 COMPLETE!")
print("="*80)
