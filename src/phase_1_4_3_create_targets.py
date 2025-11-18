"""
Phase 1.4.3 - Create Target Variables
Creates comprehensive target variables for modeling
"""
import pandas as pd
import numpy as np

def create_target_variables():
    """Create target variables for the modeling pipeline"""
    
    print("STEP 1.4.3 - CREATING TARGET VARIABLES")
    print("=" * 50)
    
    # Load the data with rankings
    print("Loading data...")
    df = pd.read_csv('data/processed/merged_with_rankings_v1.csv', low_memory=False)
    print(f"Data shape: {df.shape}")
    
    print("\nExisting target information:")
    print(f"Drafted column exists: {'Drafted' in df.columns}")
    print(f"Draft_Pick column exists: {'Draft_Pick' in df.columns}")
    
    if 'Drafted' in df.columns:
        print(f"Current draft rate: {df['Drafted'].mean():.1%}")
    
    # a) Create 'drafted' binary (0/1) based on presence of draft_pick
    print("\nCreating binary drafted target...")
    df['drafted'] = df['Draft_Pick'].notna().astype(int)
    
    # Verify consistency with existing Drafted column
    if 'Drafted' in df.columns:
        consistency = (df['drafted'] == df['Drafted']).all()
        print(f"Consistency with existing Drafted column: {consistency}")
        if not consistency:
            print("WARNING: Inconsistency found between drafted and Drafted columns")
    
    # b) Create 'draft_position' (1-60) for drafted players
    print("Creating draft position target...")
    df['draft_position'] = df['Draft_Pick'].copy()
    
    # Validate draft positions are in expected range
    drafted_mask = df['draft_position'].notna()
    if drafted_mask.sum() > 0:
        min_pick = df.loc[drafted_mask, 'draft_position'].min()
        max_pick = df.loc[drafted_mask, 'draft_position'].max()
        print(f"Draft position range: {min_pick} to {max_pick}")
        
        # Check for picks outside typical range (1-60)
        unusual_picks = df.loc[drafted_mask & ((df['draft_position'] < 1) | (df['draft_position'] > 60)), 'draft_position']
        if len(unusual_picks) > 0:
            print(f"WARNING: {len(unusual_picks)} picks outside 1-60 range: {unusual_picks.tolist()}")
    
    # c) Create 'lottery_pick' (1 if pick <= 14, else 0)
    print("Creating lottery pick indicator...")
    df['lottery_pick'] = ((df['draft_position'] <= 14) & (df['draft_position'].notna())).astype(int)
    
    # d) Create 'first_round' (1 if pick <= 30, else 0)
    print("Creating first round indicator...")
    df['first_round'] = ((df['draft_position'] <= 30) & (df['draft_position'].notna())).astype(int)
    
    # e) Create 'second_round' (1 if pick > 30, else 0)
    print("Creating second round indicator...")
    df['second_round'] = ((df['draft_position'] > 30) & (df['draft_position'].notna())).astype(int)
    
    # Additional target variables for analysis
    print("Creating additional target categories...")
    
    # Top 10 picks (elite prospects)
    df['top_10_pick'] = ((df['draft_position'] <= 10) & (df['draft_position'].notna())).astype(int)
    
    # Late first round (picks 21-30)
    df['late_first_round'] = ((df['draft_position'] >= 21) & (df['draft_position'] <= 30)).astype(int)
    
    # Draft position categories for classification
    conditions = [
        (df['draft_position'] <= 10),
        (df['draft_position'] <= 20),
        (df['draft_position'] <= 30),
        (df['draft_position'] <= 45),
        (df['draft_position'] <= 60)
    ]
    choices = ['Top_10', 'Lottery_Plus', 'First_Round', 'Early_Second', 'Late_Second']
    df['draft_tier'] = np.select(conditions, choices, default='Undrafted')
    
    # e) Validate targets have correct ranges
    print("\nValidating target variables...")
    
    validation_results = {
        'drafted': {
            'range': f"{df['drafted'].min()} to {df['drafted'].max()}",
            'expected': '0 to 1',
            'count': df['drafted'].sum()
        },
        'lottery_pick': {
            'range': f"{df['lottery_pick'].min()} to {df['lottery_pick'].max()}",
            'expected': '0 to 1', 
            'count': df['lottery_pick'].sum()
        },
        'first_round': {
            'range': f"{df['first_round'].min()} to {df['first_round'].max()}",
            'expected': '0 to 1',
            'count': df['first_round'].sum()
        },
        'second_round': {
            'range': f"{df['second_round'].min()} to {df['second_round'].max()}",
            'expected': '0 to 1',
            'count': df['second_round'].sum()
        }
    }
    
    print("Target variable validation:")
    for var, stats in validation_results.items():
        print(f"  {var}: range {stats['range']} (expected {stats['expected']}), count: {stats['count']}")
    
    # Check logical consistency
    print("\nLogical consistency checks:")
    print(f"  All lottery picks are first round: {(df['lottery_pick'] <= df['first_round']).all()}")
    print(f"  All first round are drafted: {(df['first_round'] <= df['drafted']).all()}")
    print(f"  All second round are drafted: {(df['second_round'] <= df['drafted']).all()}")
    print(f"  No overlap first/second round: {(df['first_round'] + df['second_round'] <= df['drafted']).all()}")
    
    # Summary statistics by draft year
    print("\nTarget variable summary by draft year:")
    if 'Draft_Year' in df.columns:
        draft_summary = df.groupby('Draft_Year').agg({
            'drafted': 'sum',
            'lottery_pick': 'sum', 
            'first_round': 'sum',
            'second_round': 'sum'
        }).fillna(0).astype(int)
        print(draft_summary.to_string())
    
    # Create draft tier distribution
    print(f"\nDraft tier distribution:")
    tier_counts = df['draft_tier'].value_counts()
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        print(f"  {tier}: {count:,} ({pct:.1f}%)")
    
    print(f"\nFinal data shape: {df.shape}")
    print("New target columns created:", ['drafted', 'draft_position', 'lottery_pick', 'first_round', 'second_round', 'top_10_pick', 'late_first_round', 'draft_tier'])
    
    # Save the data with target variables
    df.to_csv('data/processed/merged_with_targets_v1.csv', index=False)
    
    # Create target variables summary report
    with open('data/processed/target_variables_report.txt', 'w') as f:
        f.write("TARGET VARIABLES CREATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total records: {len(df):,}\n")
        f.write(f"Drafted players: {df['drafted'].sum():,} ({df['drafted'].mean():.1%})\n")
        f.write(f"Lottery picks: {df['lottery_pick'].sum():,} ({df['lottery_pick'].mean():.1%})\n")
        f.write(f"First round picks: {df['first_round'].sum():,} ({df['first_round'].mean():.1%})\n")
        f.write(f"Second round picks: {df['second_round'].sum():,} ({df['second_round'].mean():.1%})\n\n")
        
        f.write("DRAFT TIER DISTRIBUTION:\n")
        for tier, count in tier_counts.items():
            pct = count / len(df) * 100
            f.write(f"  {tier}: {count:,} ({pct:.1f}%)\n")
    
    print("\nStep 1.4.3 completed successfully!")
    print("Created files:")
    print("- data/processed/merged_with_targets_v1.csv")
    print("- data/processed/target_variables_report.txt")
    
    return df

if __name__ == "__main__":
    df = create_target_variables()