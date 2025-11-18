"""
Phase 1.1: Initial Data Loading and Inspection
Step 1.1.1: Load each dataset individually
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Set up paths
BASE_DIR = Path('/Users/sam/Documents/School/Emory/DataLab')
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_inspect_dataset(file_path, file_description):
    """Load a dataset and print basic information"""
    print(f"\n{'='*60}")
    print(f"Loading: {file_description}")
    print(f"File: {file_path.name}")
    print('='*60)
    
    try:
        # Load based on file extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            print(f"Unknown file type: {file_path.suffix}")
            return None
        
        print(f"✅ Successfully loaded!")
        print(f"Shape: {df.shape}")
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
        return df
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return None

# Step 1.1.1: Load each dataset individually
print("="*80)
print("STEP 1.1.1: LOAD EACH DATASET INDIVIDUALLY")
print("="*80)

# 1. Load CollegeBasketballPlayers2009-2021.csv
college_stats = load_and_inspect_dataset(
    RAW_DIR / 'CollegeBasketballPlayers2009-2021.csv',
    'College Basketball Players Statistics (2009-2021)'
)

# 2. Load DraftedPlayers2009-2021.xlsx
draft_data = load_and_inspect_dataset(
    RAW_DIR / 'DraftedPlayers2009-2021.xlsx',
    'NBA Drafted Players (2009-2021)'
)

# 3. Load Final_Year_Team_Rank.csv
team_ranks = load_and_inspect_dataset(
    RAW_DIR / 'Final_Year_Team_Rank.csv',
    'Team Rankings by Year'
)

# 4. Load modern_RAPTOR_by_player.csv
raptor_data = load_and_inspect_dataset(
    RAW_DIR / 'modern_RAPTOR_by_player.csv',
    'NBA RAPTOR Performance Metrics'
)

# 5. Load CollegePlayers_FinalYear_FULL.csv (if exists)
if (RAW_DIR / 'CollegePlayers_FinalYear_FULL.csv').exists():
    final_year_data = load_and_inspect_dataset(
        RAW_DIR / 'CollegePlayers_FinalYear_FULL.csv',
        'College Players Final Year (Full Dataset)'
    )
else:
    print("\n⚠️ CollegePlayers_FinalYear_FULL.csv not found")
    final_year_data = None

# Summary
print("\n" + "="*80)
print("SUMMARY OF LOADED DATASETS")
print("="*80)

datasets_info = {
    'college_stats': college_stats.shape if college_stats is not None else 'Not loaded',
    'draft_data': draft_data.shape if draft_data is not None else 'Not loaded',
    'team_ranks': team_ranks.shape if team_ranks is not None else 'Not loaded',
    'raptor_data': raptor_data.shape if raptor_data is not None else 'Not loaded',
    'final_year_data': final_year_data.shape if final_year_data is not None else 'Not loaded'
}

for name, shape in datasets_info.items():
    print(f"{name:20s}: {shape}")

print("\n✅ Step 1.1.1 Complete!")
