"""
Phase 1.1: Initial Data Loading and Inspection
Step 1.1.2: Create data profile report for each file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path('/Users/sam/Documents/School/Emory/DataLab')
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_data_profile(df, dataset_name, output_file):
    """Create detailed data profile report for a dataset"""
    
    with open(output_file, 'w') as f:
        f.write(f"={'='*80}\n")
        f.write(f"DATA PROFILE REPORT: {dataset_name}\n")
        f.write(f"={'='*80}\n\n")
        
        # Basic Information
        f.write(f"1. BASIC INFORMATION\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"Number of rows: {df.shape[0]:,}\n")
        f.write(f"Number of columns: {df.shape[1]}\n")
        f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        # Column Information
        f.write(f"2. COLUMN NAMES AND DATA TYPES\n")
        f.write(f"-" * 40 + "\n")
        for col, dtype in df.dtypes.items():
            f.write(f"  {col:<30} : {str(dtype):<15}\n")
        f.write(f"\n")
        
        # Data Type Summary
        f.write(f"3. DATA TYPE SUMMARY\n")
        f.write(f"-" * 40 + "\n")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            f.write(f"  {str(dtype):<15} : {count} columns\n")
        f.write(f"\n")
        
        # Missing Values
        f.write(f"4. MISSING VALUES\n")
        f.write(f"-" * 40 + "\n")
        missing_counts = df.isnull().sum()
        missing_pcts = (df.isnull().sum() / len(df)) * 100
        
        # Only show columns with missing values
        missing_df = pd.DataFrame({
            'Missing Count': missing_counts[missing_counts > 0],
            'Missing %': missing_pcts[missing_counts > 0]
        })
        
        if len(missing_df) > 0:
            missing_df = missing_df.sort_values('Missing %', ascending=False)
            for idx, row in missing_df.iterrows():
                f.write(f"  {idx:<30} : {row['Missing Count']:6.0f} ({row['Missing %']:5.1f}%)\n")
        else:
            f.write("  No missing values found!\n")
        f.write(f"\n")
        
        # Numeric Statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            f.write(f"5. NUMERIC COLUMN STATISTICS (Top 10 by variance)\n")
            f.write(f"-" * 40 + "\n")
            
            # Calculate variance and select top 10
            variances = df[numeric_cols].var().sort_values(ascending=False)
            top_numeric = variances.head(10).index
            
            stats_df = df[top_numeric].describe().T
            stats_df['variance'] = df[top_numeric].var()
            
            for col in top_numeric:
                f.write(f"\n  {col}:\n")
                f.write(f"    Mean     : {df[col].mean():10.2f}\n")
                f.write(f"    Std Dev  : {df[col].std():10.2f}\n")
                f.write(f"    Min      : {df[col].min():10.2f}\n")
                f.write(f"    25%      : {df[col].quantile(0.25):10.2f}\n")
                f.write(f"    Median   : {df[col].median():10.2f}\n")
                f.write(f"    75%      : {df[col].quantile(0.75):10.2f}\n")
                f.write(f"    Max      : {df[col].max():10.2f}\n")
        
        # Categorical Statistics
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            f.write(f"\n6. CATEGORICAL COLUMN STATISTICS\n")
            f.write(f"-" * 40 + "\n")
            for col in object_cols:
                unique_count = df[col].nunique()
                f.write(f"\n  {col}:\n")
                f.write(f"    Unique values: {unique_count}\n")
                if unique_count <= 10:
                    value_counts = df[col].value_counts().head(10)
                    for value, count in value_counts.items():
                        f.write(f"      {str(value):<20} : {count:6d} ({count/len(df)*100:5.1f}%)\n")
                else:
                    # Show top 5 values
                    value_counts = df[col].value_counts().head(5)
                    for value, count in value_counts.items():
                        f.write(f"      {str(value):<20} : {count:6d} ({count/len(df)*100:5.1f}%)\n")
                    f.write(f"      ... ({unique_count - 5} more values)\n")
        
        # Sample Data
        f.write(f"\n7. SAMPLE DATA (First 5 rows)\n")
        f.write(f"-" * 40 + "\n")
        sample_str = df.head(5).to_string()
        f.write(sample_str + "\n")
        
        print(f"✅ Data profile saved to: {output_file}")
        return output_file

# Load and profile each dataset
print("="*80)
print("STEP 1.1.2: CREATE DATA PROFILE REPORTS")
print("="*80)

# 1. College Basketball Players
print("\n📊 Profiling: CollegeBasketballPlayers2009-2021.csv")
college_stats = pd.read_csv(RAW_DIR / 'CollegeBasketballPlayers2009-2021.csv', low_memory=False)
create_data_profile(
    college_stats,
    "College Basketball Players 2009-2021",
    RESULTS_DIR / "data_profile_college_stats.txt"
)

# 2. Drafted Players
print("\n📊 Profiling: DraftedPlayers2009-2021.xlsx")
draft_data = pd.read_excel(RAW_DIR / 'DraftedPlayers2009-2021.xlsx')
create_data_profile(
    draft_data,
    "NBA Drafted Players 2009-2021",
    RESULTS_DIR / "data_profile_draft_data.txt"
)

# 3. Team Rankings
print("\n📊 Profiling: Final_Year_Team_Rank.csv")
team_ranks = pd.read_csv(RAW_DIR / 'Final_Year_Team_Rank.csv')
create_data_profile(
    team_ranks,
    "Team Rankings by Year",
    RESULTS_DIR / "data_profile_team_ranks.txt"
)

# 4. RAPTOR Data
print("\n📊 Profiling: modern_RAPTOR_by_player.csv")
raptor_data = pd.read_csv(RAW_DIR / 'modern_RAPTOR_by_player.csv')
create_data_profile(
    raptor_data,
    "NBA RAPTOR Performance Metrics",
    RESULTS_DIR / "data_profile_raptor.txt"
)

# 5. Final Year Data
print("\n📊 Profiling: CollegePlayers_FinalYear_FULL.csv")
final_year_data = pd.read_csv(RAW_DIR / 'CollegePlayers_FinalYear_FULL.csv', low_memory=False)
create_data_profile(
    final_year_data,
    "College Players Final Year Full",
    RESULTS_DIR / "data_profile_final_year.txt"
)

print("\n" + "="*80)
print("✅ STEP 1.1.2 COMPLETE!")
print("All data profiles saved to results/ directory")
print("="*80)
