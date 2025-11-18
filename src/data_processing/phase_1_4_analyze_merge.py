"""
Phase 1.4 - Analyze Current Merge Quality
Analyzes the existing merged_raw_v1.csv to understand merge quality before proceeding
"""
import pandas as pd
import numpy as np

def analyze_merge_quality():
    """Analyze the quality of the current merge"""
    
    # Load the merged data
    merged_df = pd.read_csv('data/processed/merged_raw_v1.csv')
    
    print("MERGE QUALITY ANALYSIS")
    print("=" * 50)
    print(f"Total records: {len(merged_df):,}")
    print(f"Total drafted players: {merged_df['Drafted'].sum():,}")
    print(f"Draft rate: {merged_df['Drafted'].mean():.1%}")
    print()
    
    # Check draft years coverage
    print("DRAFT YEARS COVERAGE:")
    draft_years = merged_df[merged_df['Drafted'] == 1]['Draft_Year'].value_counts().sort_index()
    print(draft_years)
    print()
    
    # Check missing pick information
    drafted_players = merged_df[merged_df['Drafted'] == 1]
    missing_picks = drafted_players['Draft_Pick'].isna().sum()
    print(f"Drafted players missing pick information: {missing_picks}")
    
    # Sample of drafted players
    print("\nSAMPLE DRAFTED PLAYERS:")
    sample_drafted = drafted_players[['Player', 'Season', 'Draft_Year', 'Draft_Pick', 'NBA_Team']].head()
    print(sample_drafted.to_string(index=False))
    
    return merged_df

if __name__ == "__main__":
    df = analyze_merge_quality()