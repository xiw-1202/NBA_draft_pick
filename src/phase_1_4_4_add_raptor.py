"""
Phase 1.4.4 - Add RAPTOR Metrics for Diamond Analysis
Merges NBA RAPTOR performance data to identify diamond players
"""
import pandas as pd
import numpy as np

def clean_player_name_for_matching(name):
    """Clean player name for better matching"""
    if pd.isna(name):
        return name
    
    # Convert to string and strip whitespace
    name = str(name).strip()
    
    # Remove common suffixes
    suffixes = [' Jr.', ' Sr.', ' III', ' II', ' IV']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Handle common name variations
    name_replacements = {
        'Charles ': 'Charlie ',
        'Robert ': 'Bob ',
        'William ': 'Bill ',
        'Richard ': 'Rick ',
        'Michael ': 'Mike ',
        'Christopher ': 'Chris ',
        'Matthew ': 'Matt ',
        'Joseph ': 'Joe ',
        'Anthony ': 'Tony '
    }
    
    for old, new in name_replacements.items():
        if name.startswith(old):
            name = new + name[len(old):]
            break
    
    return name

def add_raptor_metrics():
    """Add RAPTOR metrics for diamond player analysis"""
    
    print("STEP 1.4.4 - ADDING RAPTOR METRICS")
    print("=" * 50)
    
    # Load datasets
    print("Loading data...")
    merged_df = pd.read_csv('data/processed/merged_with_targets_v1.csv', low_memory=False)
    raptor_df = pd.read_csv('data/raw/modern_RAPTOR_by_player.csv')
    
    print(f"Merged data shape: {merged_df.shape}")
    print(f"RAPTOR data shape: {raptor_df.shape}")
    print(f"RAPTOR seasons covered: {raptor_df['season'].min()}-{raptor_df['season'].max()}")
    
    # Clean player names for matching
    print("\nCleaning player names for matching...")
    merged_df['player_name_clean'] = merged_df['Player'].apply(clean_player_name_for_matching)
    raptor_df['player_name_clean'] = raptor_df['player_name'].apply(clean_player_name_for_matching)
    
    # Calculate career average RAPTOR metrics
    print("Calculating career average RAPTOR metrics...")
    raptor_career = raptor_df.groupby('player_name_clean').agg({
        'poss': 'sum',  # Total possessions
        'mp': 'sum',    # Total minutes
        'raptor_offense': 'mean',  # Average offensive RAPTOR
        'raptor_defense': 'mean',  # Average defensive RAPTOR  
        'raptor_total': 'mean',    # Average total RAPTOR
        'war_total': 'sum',        # Total WAR across career
        'season': ['min', 'max', 'count']  # Career span info
    }).round(3)
    
    # Flatten column names
    raptor_career.columns = ['total_poss', 'total_mp', 'raptor_offense_avg', 'raptor_defense_avg', 
                           'raptor_total_avg', 'war_total_career', 'first_nba_season', 'last_nba_season', 'nba_seasons']
    raptor_career = raptor_career.reset_index()
    
    # Filter for players with meaningful NBA careers (>500 total minutes)
    raptor_career = raptor_career[raptor_career['total_mp'] >= 500].copy()
    print(f"NBA players with >500 career minutes: {len(raptor_career)}")
    
    # Match player names between datasets
    print("Matching players between datasets...")
    
    # Direct matches first
    college_players = set(merged_df['player_name_clean'].dropna())
    nba_players = set(raptor_career['player_name_clean'])
    direct_matches = college_players.intersection(nba_players)
    print(f"Direct name matches: {len(direct_matches)}")
    
    # Sample of matched players
    print("\nSample matched players:")
    sample_matched = list(direct_matches)[:10]
    for player in sample_matched:
        print(f"  {player}")
    
    # Merge RAPTOR data
    print(f"\nMerging RAPTOR data...")
    merged_with_raptor = merged_df.merge(
        raptor_career, 
        on='player_name_clean', 
        how='left'
    )
    
    # Create RAPTOR availability flag
    merged_with_raptor['has_nba_data'] = merged_with_raptor['raptor_total_avg'].notna().astype(int)
    
    # Define diamond player criteria
    print("Defining diamond player criteria...")
    
    # Diamond criteria: High NBA performance (RAPTOR > 1.0) AND either undrafted or late draft pick (>40)
    high_raptor_threshold = 1.0
    late_pick_threshold = 40
    
    # Create diamond flags
    conditions_diamond = (
        (merged_with_raptor['raptor_total_avg'] > high_raptor_threshold) &  # High NBA performance
        ((merged_with_raptor['drafted'] == 0) |  # Undrafted OR
         (merged_with_raptor['draft_position'] > late_pick_threshold))  # Late pick
    )
    
    merged_with_raptor['is_diamond'] = conditions_diamond.astype(int)
    
    # Create additional diamond categories
    merged_with_raptor['undrafted_diamond'] = (
        (merged_with_raptor['drafted'] == 0) & 
        (merged_with_raptor['raptor_total_avg'] > high_raptor_threshold)
    ).astype(int)
    
    merged_with_raptor['late_pick_diamond'] = (
        (merged_with_raptor['draft_position'] > late_pick_threshold) & 
        (merged_with_raptor['raptor_total_avg'] > high_raptor_threshold)
    ).astype(int)
    
    # High value players (very high RAPTOR, any draft position)
    merged_with_raptor['elite_nba_player'] = (merged_with_raptor['raptor_total_avg'] > 2.0).astype(int)
    
    # Calculate statistics
    print("\nRAPTOR merge statistics:")
    print(f"Players with NBA data: {merged_with_raptor['has_nba_data'].sum():,} ({merged_with_raptor['has_nba_data'].mean():.1%})")
    print(f"Diamond players: {merged_with_raptor['is_diamond'].sum():,}")
    print(f"Undrafted diamonds: {merged_with_raptor['undrafted_diamond'].sum():,}")
    print(f"Late pick diamonds: {merged_with_raptor['late_pick_diamond'].sum():,}")
    print(f"Elite NBA players: {merged_with_raptor['elite_nba_player'].sum():,}")
    
    # Show diamond players
    diamond_players = merged_with_raptor[merged_with_raptor['is_diamond'] == 1][
        ['Player', 'Season', 'Team', 'drafted', 'draft_position', 'raptor_total_avg', 'war_total_career']
    ].sort_values('raptor_total_avg', ascending=False)
    
    print(f"\nIdentified Diamond Players ({len(diamond_players)}):")
    if len(diamond_players) > 0:
        print(diamond_players.head(10).to_string(index=False))
    
    # Show elite undrafted players
    undrafted_elite = merged_with_raptor[
        (merged_with_raptor['drafted'] == 0) & 
        (merged_with_raptor['raptor_total_avg'].notna())
    ][['Player', 'Season', 'Team', 'raptor_total_avg', 'war_total_career']].sort_values('raptor_total_avg', ascending=False)
    
    print(f"\nTop Undrafted Players with NBA Data ({len(undrafted_elite)}):")
    if len(undrafted_elite) > 0:
        print(undrafted_elite.head(10).to_string(index=False))
    
    # Draft position vs NBA performance analysis
    drafted_with_raptor = merged_with_raptor[
        (merged_with_raptor['drafted'] == 1) & 
        (merged_with_raptor['has_nba_data'] == 1)
    ]
    
    if len(drafted_with_raptor) > 0:
        print(f"\nDraft position vs NBA performance correlation:")
        corr = drafted_with_raptor[['draft_position', 'raptor_total_avg']].corr().iloc[0,1]
        print(f"Correlation (draft_position vs raptor_total_avg): {corr:.3f}")
        
        # Average RAPTOR by draft round
        print("\nAverage RAPTOR by draft tier:")
        raptor_by_tier = drafted_with_raptor.groupby('draft_tier')['raptor_total_avg'].agg(['count', 'mean']).round(3)
        print(raptor_by_tier.to_string())
    
    print(f"\nFinal data shape: {merged_with_raptor.shape}")
    
    # Save the enhanced dataset
    merged_with_raptor.to_csv('data/processed/merged_with_raptor_v1.csv', index=False)
    
    # Create RAPTOR analysis report
    with open('data/processed/raptor_analysis_report.txt', 'w') as f:
        f.write("RAPTOR METRICS ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total college players: {len(merged_with_raptor):,}\n")
        f.write(f"Players with NBA data: {merged_with_raptor['has_nba_data'].sum():,} ({merged_with_raptor['has_nba_data'].mean():.1%})\n")
        f.write(f"Diamond players identified: {merged_with_raptor['is_diamond'].sum():,}\n")
        f.write(f"Undrafted diamonds: {merged_with_raptor['undrafted_diamond'].sum():,}\n")
        f.write(f"Late pick diamonds: {merged_with_raptor['late_pick_diamond'].sum():,}\n")
        f.write(f"Elite NBA players: {merged_with_raptor['elite_nba_player'].sum():,}\n\n")
        
        f.write("DIAMOND PLAYER CRITERIA:\n")
        f.write(f"- High NBA performance: RAPTOR > {high_raptor_threshold}\n")
        f.write(f"- AND (Undrafted OR drafted after pick {late_pick_threshold})\n\n")
        
        if len(diamond_players) > 0:
            f.write("IDENTIFIED DIAMOND PLAYERS:\n")
            f.write(diamond_players.to_string(index=False))
            f.write("\n\n")
        
        if len(drafted_with_raptor) > 0:
            f.write(f"Draft position vs NBA performance correlation: {corr:.3f}\n")
    
    print("\nStep 1.4.4 completed successfully!")
    print("Created files:")
    print("- data/processed/merged_with_raptor_v1.csv")
    print("- data/processed/raptor_analysis_report.txt")
    
    return merged_with_raptor

if __name__ == "__main__":
    df = add_raptor_metrics()