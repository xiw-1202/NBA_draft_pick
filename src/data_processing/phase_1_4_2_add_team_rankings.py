"""
Phase 1.4.2 - Add Team Rankings to Merged Data
Merges team rankings with the existing merged dataset
"""
import pandas as pd
import numpy as np

def add_team_rankings():
    """Add team rankings to the merged dataset"""
    
    print("STEP 1.4.2 - ADDING TEAM RANKINGS")
    print("=" * 50)
    
    # Load datasets
    print("Loading data...")
    merged_df = pd.read_csv('data/processed/merged_raw_v1.csv', low_memory=False)
    team_ranks_df = pd.read_csv('data/processed/team_ranks_long.csv')
    
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Team rankings shape: {team_ranks_df.shape}")
    
    # Create team_year_key for merging
    print("\nCreating merge keys...")
    merged_df['team_year_key'] = merged_df['Team'] + '_' + merged_df['Season'].astype(str)
    team_ranks_df['team_year_key'] = team_ranks_df['team'] + '_' + team_ranks_df['year'].astype(str)
    
    print(f"Unique team-year combinations in merged data: {merged_df['team_year_key'].nunique()}")
    print(f"Unique team-year combinations in rankings: {team_ranks_df['team_year_key'].nunique()}")
    
    # Check for direct matches
    direct_matches = set(merged_df['team_year_key']).intersection(set(team_ranks_df['team_year_key']))
    print(f"Direct matches: {len(direct_matches)}")
    
    # Sample unmatched teams to see name differences
    unmatched_merged = set(merged_df['team_year_key']) - set(team_ranks_df['team_year_key'])
    print(f"Unmatched from merged data: {len(unmatched_merged)}")
    print("Sample unmatched teams:")
    sample_unmatched = list(unmatched_merged)[:10]
    for team_year in sample_unmatched:
        print(f"  {team_year}")
    
    # For now, do a simple merge and fill missing with defaults
    print("\nPerforming merge...")
    merged_with_ranks = merged_df.merge(
        team_ranks_df[['team_year_key', 'ap_rank', 'kenpom_rank']], 
        on='team_year_key', 
        how='left'
    )
    
    # Fill missing rankings with default values (999 = unranked)
    merged_with_ranks['ap_rank'] = merged_with_ranks['ap_rank'].fillna(999)
    merged_with_ranks['kenpom_rank'] = merged_with_ranks['kenpom_rank'].fillna(999)
    
    # Create ranking indicator features
    merged_with_ranks['has_ap_ranking'] = (merged_with_ranks['ap_rank'] < 999).astype(int)
    merged_with_ranks['has_kenpom_ranking'] = (merged_with_ranks['kenpom_rank'] < 999).astype(int)
    merged_with_ranks['top_25_team'] = (merged_with_ranks['ap_rank'] <= 25).astype(int)
    
    # Calculate ranking statistics
    print("\nRanking coverage statistics:")
    print(f"Teams with AP ranking: {merged_with_ranks['has_ap_ranking'].sum():,} ({merged_with_ranks['has_ap_ranking'].mean():.1%})")
    print(f"Teams with KenPom ranking: {merged_with_ranks['has_kenpom_ranking'].sum():,} ({merged_with_ranks['has_kenpom_ranking'].mean():.1%})")
    print(f"Top 25 teams: {merged_with_ranks['top_25_team'].sum():,} ({merged_with_ranks['top_25_team'].mean():.1%})")
    
    # Sample of teams with rankings
    print("\nSample teams with rankings:")
    ranked_sample = merged_with_ranks[merged_with_ranks['has_ap_ranking'] == 1][
        ['Player', 'Team', 'Season', 'ap_rank', 'kenpom_rank']].head()
    print(ranked_sample.to_string(index=False))
    
    # Drop temporary merge key
    merged_with_ranks = merged_with_ranks.drop('team_year_key', axis=1)
    
    # Save updated merged data
    print(f"\nSaving updated merged data with shape: {merged_with_ranks.shape}")
    merged_with_ranks.to_csv('data/processed/merged_with_rankings_v1.csv', index=False)
    
    # Create summary report
    with open('data/processed/rankings_merge_report.txt', 'w') as f:
        f.write("TEAM RANKINGS MERGE REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total records: {len(merged_with_ranks):,}\n")
        f.write(f"Records with AP ranking: {merged_with_ranks['has_ap_ranking'].sum():,} ({merged_with_ranks['has_ap_ranking'].mean():.1%})\n")
        f.write(f"Records with KenPom ranking: {merged_with_ranks['has_kenpom_ranking'].sum():,} ({merged_with_ranks['has_kenpom_ranking'].mean():.1%})\n")
        f.write(f"Top 25 teams: {merged_with_ranks['top_25_team'].sum():,} ({merged_with_ranks['top_25_team'].mean():.1%})\n")
        f.write(f"Direct team name matches: {len(direct_matches)}\n")
        f.write(f"Unmatched team-year combinations: {len(unmatched_merged)}\n")
    
    print("\nStep 1.4.2 completed successfully!")
    print("Created files:")
    print("- data/processed/merged_with_rankings_v1.csv")
    print("- data/processed/rankings_merge_report.txt")
    
    return merged_with_ranks

if __name__ == "__main__":
    df = add_team_rankings()