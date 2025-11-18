"""
Phase 1.4.5 - Final Merge Validation and Output
Comprehensive validation of the complete merged dataset
"""
import pandas as pd
import numpy as np

def final_merge_validation():
    """Perform comprehensive validation of the merged dataset"""
    
    print("STEP 1.4.5 - FINAL MERGE VALIDATION")
    print("=" * 60)
    
    # Load final dataset
    print("Loading final merged dataset...")
    df = pd.read_csv('data/processed/merged_with_raptor_v1.csv', low_memory=False)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Total records: {len(df):,}")
    
    # a) Check for duplicate rows
    print("\n1. DUPLICATE ROW ANALYSIS")
    print("-" * 30)
    duplicates = df.duplicated()
    print(f"Duplicate rows: {duplicates.sum()}")
    
    # Check for duplicate player-season combinations (which should be unique)
    if 'Player' in df.columns and 'Season' in df.columns:
        player_season_dupes = df.duplicated(subset=['Player', 'Season'])
        print(f"Duplicate player-season combinations: {player_season_dupes.sum()}")
        
        if player_season_dupes.sum() > 0:
            print("Sample duplicate player-seasons:")
            dupe_examples = df[player_season_dupes][['Player', 'Season', 'Team']].head()
            print(dupe_examples.to_string(index=False))
    
    # b) Verify all expected columns are present
    print("\n2. COLUMN VERIFICATION")
    print("-" * 30)
    expected_column_groups = {
        'identifiers': ['Player', 'Season', 'Team'],
        'college_stats': ['GP', 'Min_per', 'pts', 'ast', 'treb'],  # Sample key stats
        'draft_info': ['drafted', 'draft_position', 'Draft_Year', 'NBA_Team'],
        'targets': ['lottery_pick', 'first_round', 'second_round', 'draft_tier'],
        'team_rankings': ['ap_rank', 'kenpom_rank', 'has_ap_ranking', 'top_25_team'],
        'raptor_metrics': ['has_nba_data', 'raptor_total_avg', 'is_diamond'],
    }
    
    missing_columns = []
    for group, cols in expected_column_groups.items():
        group_missing = [col for col in cols if col not in df.columns]
        if group_missing:
            missing_columns.extend(group_missing)
            print(f"Missing {group} columns: {group_missing}")
        else:
            print(f"✓ All {group} columns present")
    
    if not missing_columns:
        print("✓ All expected columns are present")
    
    # c) Generate merge statistics report
    print("\n3. MERGE STATISTICS REPORT")
    print("-" * 30)
    
    # Draft statistics
    total_drafted = df['drafted'].sum() if 'drafted' in df.columns else 0
    draft_rate = df['drafted'].mean() if 'drafted' in df.columns else 0
    
    print(f"Total college player records: {len(df):,}")
    print(f"Total drafted players: {total_drafted:,} ({draft_rate:.1%})")
    
    if 'Draft_Year' in df.columns:
        draft_years = df[df['drafted'] == 1]['Draft_Year'].value_counts().sort_index()
        print(f"Draft years covered: {draft_years.index.min():.0f}-{draft_years.index.max():.0f}")
        print(f"Average drafts per year: {draft_years.mean():.1f}")
    
    # Team ranking statistics
    if 'has_ap_ranking' in df.columns:
        ranked_teams = df['has_ap_ranking'].sum()
        ranking_rate = df['has_ap_ranking'].mean()
        print(f"Player-seasons from ranked teams: {ranked_teams:,} ({ranking_rate:.1%})")
    
    # RAPTOR statistics
    if 'has_nba_data' in df.columns:
        nba_players = df['has_nba_data'].sum()
        nba_rate = df['has_nba_data'].mean()
        print(f"Players with NBA performance data: {nba_players:,} ({nba_rate:.1%})")
        
        if 'is_diamond' in df.columns:
            diamonds = df['is_diamond'].sum()
            print(f"Diamond players identified: {diamonds:,}")
    
    # d) Verify no infinite or NaN values in critical columns
    print("\n4. DATA QUALITY CHECKS")
    print("-" * 30)
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print("Columns with infinite values:")
        for col, count in inf_counts.items():
            print(f"  {col}: {count}")
    else:
        print("✓ No infinite values found")
    
    # Check for missing values in critical columns
    critical_cols = ['Player', 'Season', 'Team', 'GP', 'drafted']
    critical_missing = {}
    for col in critical_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                critical_missing[col] = missing_count
    
    if critical_missing:
        print("Missing values in critical columns:")
        for col, count in critical_missing.items():
            print(f"  {col}: {count}")
    else:
        print("✓ No missing values in critical columns")
    
    # e) Create data dictionary with feature descriptions
    print("\n5. CREATING DATA DICTIONARY")
    print("-" * 30)
    
    # Create comprehensive data dictionary
    data_dict = {
        # Player identifiers
        'Player': 'Player name',
        'Season': 'College season year', 
        'Team': 'College team name',
        'Conference': 'College conference',
        
        # College performance stats
        'GP': 'Games played',
        'Min_per': 'Minutes per game',
        'pts': 'Points per game',
        'ast': 'Assists per game', 
        'treb': 'Total rebounds per game',
        'eFG': 'Effective field goal percentage',
        'TS_per': 'True shooting percentage',
        'usg': 'Usage rate',
        
        # Draft information
        'drafted': 'Binary indicator if player was drafted (0/1)',
        'draft_position': 'Draft position (1-60) for drafted players',
        'Draft_Year': 'Year player was drafted',
        'NBA_Team': 'NBA team that drafted the player',
        
        # Target variables
        'lottery_pick': 'Binary indicator for lottery pick (picks 1-14)',
        'first_round': 'Binary indicator for first round pick (picks 1-30)',
        'second_round': 'Binary indicator for second round pick (picks 31-60)',
        'draft_tier': 'Categorical draft tier (Top_10, Lottery_Plus, First_Round, etc.)',
        
        # Team context
        'ap_rank': 'AP poll ranking (999 if unranked)',
        'kenpom_rank': 'KenPom ranking (999 if unranked)',
        'has_ap_ranking': 'Binary indicator if team was AP ranked',
        'top_25_team': 'Binary indicator if team was top 25 AP ranked',
        
        # NBA performance (for diamond analysis)
        'has_nba_data': 'Binary indicator if player has NBA RAPTOR data',
        'raptor_total_avg': 'Career average total RAPTOR score',
        'war_total_career': 'Career total Wins Above Replacement',
        'is_diamond': 'Binary indicator for diamond player (high NBA performance, late/undrafted)',
        'undrafted_diamond': 'Binary indicator for undrafted diamond player',
        'elite_nba_player': 'Binary indicator for elite NBA performer (RAPTOR > 2.0)'
    }
    
    # Save data dictionary
    with open('data/final/data_dictionary.txt', 'w') as f:
        f.write("NBA DRAFT PREDICTION DATASET - DATA DICTIONARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: merged_dataset_final.csv\n")
        f.write(f"Records: {len(df):,}\n") 
        f.write(f"Features: {len(df.columns)}\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for col, desc in data_dict.items():
            if col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isna().sum()
                missing_pct = missing / len(df) * 100
                f.write(f"{col:25} | {dtype:12} | {desc:50} | Missing: {missing:5} ({missing_pct:4.1f}%)\n")
    
    # f) Save feature list for modeling
    print("Saving feature metadata...")
    
    # Identify feature columns (exclude identifiers and targets)
    identifier_cols = ['Player', 'Season', 'Team', 'Conference', 'NBA_Team', 'Draft_Year', 'player_name_clean']
    target_cols = ['drafted', 'draft_position', 'lottery_pick', 'first_round', 'second_round', 
                  'draft_tier', 'top_10_pick', 'late_first_round']
    analysis_cols = ['has_nba_data', 'raptor_total_avg', 'war_total_career', 'is_diamond', 
                    'undrafted_diamond', 'elite_nba_player']
    
    exclude_cols = identifier_cols + target_cols + analysis_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Save feature lists
    with open('data/final/feature_lists.txt', 'w') as f:
        f.write("NBA DRAFT PREDICTION - FEATURE LISTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("IDENTIFIER COLUMNS:\n")
        for col in identifier_cols:
            if col in df.columns:
                f.write(f"  {col}\n")
        
        f.write(f"\nTARGET COLUMNS ({len([c for c in target_cols if c in df.columns])}):\n")
        for col in target_cols:
            if col in df.columns:
                f.write(f"  {col}\n")
                
        f.write(f"\nFEATURE COLUMNS ({len(feature_cols)}):\n")
        for col in feature_cols:
            f.write(f"  {col}\n")
            
        f.write(f"\nANALYSIS COLUMNS ({len([c for c in analysis_cols if c in df.columns])}):\n")
        for col in analysis_cols:
            if col in df.columns:
                f.write(f"  {col}\n")
    
    # g) Save as 'merged_dataset_final.csv'
    print("Saving final dataset...")
    df.to_csv('data/final/merged_dataset_final.csv', index=False)
    
    # h) Create 'merge_log.txt' with all issues found
    print("Creating merge log...")
    with open('data/final/merge_log.txt', 'w') as f:
        f.write("NBA DRAFT PREDICTION - MERGE LOG\n")
        f.write("=" * 50 + "\n")
        f.write(f"Merge completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PHASE 1.4 COMPLETION SUMMARY:\n")
        f.write("✓ Step 1.4.1 - College stats merged with draft data\n")
        f.write("✓ Step 1.4.2 - Team rankings added\n")
        f.write("✓ Step 1.4.3 - Target variables created\n")
        f.write("✓ Step 1.4.4 - RAPTOR metrics added for diamond analysis\n")
        f.write("✓ Step 1.4.5 - Final validation completed\n\n")
        
        f.write("DATA QUALITY ISSUES:\n")
        if duplicates.sum() > 0:
            f.write(f"- {duplicates.sum()} duplicate rows found\n")
        if missing_columns:
            f.write(f"- Missing expected columns: {missing_columns}\n")
        if inf_counts:
            f.write(f"- Infinite values found in: {list(inf_counts.keys())}\n")
        if critical_missing:
            f.write(f"- Missing values in critical columns: {list(critical_missing.keys())}\n")
        
        if not any([duplicates.sum() > 0, missing_columns, inf_counts, critical_missing]):
            f.write("- No critical data quality issues found\n")
        
        f.write(f"\nFINAL STATISTICS:\n")
        f.write(f"- Total records: {len(df):,}\n")
        f.write(f"- Total features: {len(df.columns)}\n")
        f.write(f"- Drafted players: {total_drafted:,} ({draft_rate:.1%})\n")
        if 'has_nba_data' in df.columns:
            f.write(f"- Players with NBA data: {df['has_nba_data'].sum():,}\n")
        if 'is_diamond' in df.columns:
            f.write(f"- Diamond players: {df['is_diamond'].sum():,}\n")
    
    print("\n" + "="*60)
    print("PHASE 1.4 - MERGING DATASETS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFinal outputs created:")
    print("✓ data/final/merged_dataset_final.csv - Complete merged dataset")
    print("✓ data/final/data_dictionary.txt - Feature descriptions")
    print("✓ data/final/feature_lists.txt - Organized feature lists")  
    print("✓ data/final/merge_log.txt - Merge completion log")
    print()
    print(f"Dataset ready for Phase 2: Feature Engineering")
    print(f"Records: {len(df):,} | Features: {len(df.columns)} | Drafted: {total_drafted:,}")
    
    # Return key statistics for verification
    return {
        'total_records': len(df),
        'total_features': len(df.columns),
        'drafted_players': total_drafted,
        'draft_rate': draft_rate,
        'nba_players': df['has_nba_data'].sum() if 'has_nba_data' in df.columns else 0,
        'diamond_players': df['is_diamond'].sum() if 'is_diamond' in df.columns else 0,
        'data_quality_issues': len(missing_columns) + len(inf_counts) + len(critical_missing) + duplicates.sum()
    }

if __name__ == "__main__":
    stats = final_merge_validation()