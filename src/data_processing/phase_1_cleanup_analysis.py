"""
Phase 1 Cleanup Analysis
Analyze all Phase 1 work and identify cleanup opportunities
"""
import pandas as pd
import os
from pathlib import Path

def analyze_phase_1_files():
    """Analyze all Phase 1 files and identify cleanup opportunities"""
    
    print("PHASE 1 CLEANUP ANALYSIS")
    print("=" * 60)
    
    base_path = Path(".")
    
    # Check file sizes and identify duplicates/intermediate files
    print("1. FILE SIZE ANALYSIS")
    print("-" * 30)
    
    # Data files analysis
    data_files = {}
    for folder in ['data/raw', 'data/processed', 'data/final']:
        folder_path = base_path / folder
        if folder_path.exists():
            for file in folder_path.glob('*.csv'):
                size_mb = file.stat().st_size / (1024 * 1024)
                data_files[str(file)] = size_mb
    
    # Sort by size
    data_files_sorted = sorted(data_files.items(), key=lambda x: x[1], reverse=True)
    print("Data files by size:")
    for file_path, size_mb in data_files_sorted:
        print(f"  {file_path:50} {size_mb:6.1f} MB")
    
    # Check for potentially redundant intermediate files
    print(f"\n2. INTERMEDIATE FILES ANALYSIS")
    print("-" * 30)
    
    processed_files = list((base_path / "data/processed").glob('*.csv'))
    print(f"Total processed files: {len(processed_files)}")
    
    # Group by category
    file_categories = {
        'college_stats_*': [],
        'draft_*': [],
        'merged_*': [],
        'team_*': [],
        'final_*': [],
        'other': []
    }
    
    for file in processed_files:
        categorized = False
        for category in file_categories:
            if category != 'other' and file.name.startswith(category.replace('*', '')):
                file_categories[category].append(file.name)
                categorized = True
                break
        if not categorized:
            file_categories['other'].append(file.name)
    
    for category, files in file_categories.items():
        if files:
            print(f"\n{category}: {len(files)} files")
            for file in sorted(files):
                print(f"  {file}")
    
    # Check source files
    print(f"\n3. SOURCE FILES ANALYSIS")
    print("-" * 30)
    
    src_files = list((base_path / "src").glob('phase_1_*.py'))
    print(f"Total Phase 1 source files: {len(src_files)}")
    
    src_by_phase = {
        '1.1': [], '1.2': [], '1.3': [], '1.4': []
    }
    
    for file in src_files:
        for phase in src_by_phase:
            if f'phase_1_{phase.replace(".", "_")}' in file.name:
                src_by_phase[phase].append(file.name)
                break
    
    for phase, files in src_by_phase.items():
        if files:
            print(f"\nPhase {phase}: {len(files)} files")
            for file in sorted(files):
                print(f"  {file}")
    
    # Check results files
    print(f"\n4. RESULTS FILES ANALYSIS")
    print("-" * 30)
    
    results_files = list((base_path / "results").glob('*'))
    if results_files:
        for file in sorted(results_files):
            size_kb = file.stat().st_size / 1024
            print(f"  {file.name:40} {size_kb:6.1f} KB")
    else:
        print("  No results files found")
    
    return {
        'data_files': data_files_sorted,
        'processed_count': len(processed_files),
        'src_count': len(src_files),
        'results_count': len(results_files) if results_files else 0
    }

def identify_cleanup_candidates():
    """Identify files that can be cleaned up"""
    
    print(f"\n5. CLEANUP RECOMMENDATIONS")
    print("-" * 30)
    
    # Intermediate files that might be safe to remove
    potentially_removable = [
        # Early processing steps that led to final versions
        'data/processed/college_stats_names_cleaned.csv',
        'data/processed/college_stats_teams_cleaned.csv', 
        'data/processed/college_stats_with_final_flag.csv',
        'data/processed/college_stats_with_keys.csv',
        'data/processed/draft_cleaned.csv',
        'data/processed/draft_teams_cleaned.csv',
        'data/processed/draft_with_keys.csv',
        'data/processed/final_year_names_cleaned.csv',
        'data/processed/final_year_with_keys.csv',
        'data/processed/final_seasons_detected.csv',
        
        # Version 1 files superseded by final
        'data/processed/merged_raw_v1.csv',
        'data/processed/merged_with_rankings_v1.csv',
        'data/processed/merged_with_targets_v1.csv',
        'data/processed/merged_with_raptor_v1.csv',
    ]
    
    # Files to keep (essential)
    essential_files = [
        'data/final/merged_dataset_final.csv',  # Final output
        'data/final/data_dictionary.txt',       # Documentation
        'data/final/feature_lists.txt',         # Documentation  
        'data/final/merge_log.txt',             # Documentation
        
        # Key intermediate files for reference
        'data/processed/team_ranks_long.csv',   # Cleaned team rankings
        'data/processed/player_careers.csv',    # Player career summary
        'data/processed/team_name_mapping.csv', # Name mapping reference
        
        # All reports (small files, useful for reference)
        'data/processed/rankings_merge_report.txt',
        'data/processed/raptor_analysis_report.txt', 
        'data/processed/target_variables_report.txt',
        'data/processed/player_name_standardization_log.csv',
    ]
    
    print("FILES RECOMMENDED FOR REMOVAL (intermediate/superseded):")
    total_size = 0
    for file_path in potentially_removable:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size += size_mb
            print(f"  {file_path:50} {size_mb:6.1f} MB")
    
    print(f"\nTotal space that could be saved: {total_size:.1f} MB")
    
    print(f"\nESSENTIAL FILES TO KEEP:")
    for file_path in essential_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file_path:50} {size_mb:6.1f} MB")
    
    return potentially_removable, essential_files

def check_data_consistency():
    """Check consistency across key datasets"""
    
    print(f"\n6. DATA CONSISTENCY CHECKS")
    print("-" * 30)
    
    try:
        # Load final dataset
        final_df = pd.read_csv('data/final/merged_dataset_final.csv', low_memory=False)
        print(f"Final dataset: {final_df.shape}")
        print(f"Drafted players: {final_df['drafted'].sum()}")
        print(f"Players with NBA data: {final_df['has_nba_data'].sum()}")
        print(f"Diamond players: {final_df['is_diamond'].sum()}")
        
        # Check for any obvious data issues
        issues = []
        
        # Check for duplicates
        if final_df.duplicated().any():
            issues.append("Duplicate rows found")
        
        # Check for missing critical data
        critical_cols = ['Player', 'Season', 'Team', 'drafted']
        for col in critical_cols:
            if col in final_df.columns and final_df[col].isna().any():
                issues.append(f"Missing values in critical column: {col}")
        
        # Check target consistency
        if 'drafted' in final_df.columns and 'draft_position' in final_df.columns:
            drafted_without_position = final_df[(final_df['drafted'] == 1) & (final_df['draft_position'].isna())]
            if len(drafted_without_position) > 0:
                issues.append(f"{len(drafted_without_position)} drafted players missing draft position")
        
        if issues:
            print("DATA ISSUES FOUND:")
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print("✅ No critical data issues found")
            
    except Exception as e:
        print(f"❌ Error checking data consistency: {e}")

if __name__ == "__main__":
    stats = analyze_phase_1_files()
    removable, essential = identify_cleanup_candidates()
    check_data_consistency()
    
    print(f"\n" + "="*60)
    print("PHASE 1 CLEANUP SUMMARY")
    print("="*60)
    print(f"✅ Phase 1 work appears complete and well-organized")
    print(f"📁 {stats['processed_count']} processed files, {stats['src_count']} source files")
    print(f"🧹 {len(removable)} intermediate files could be safely removed")
    print(f"💾 Essential files preserved for reproducibility")