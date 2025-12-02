"""
NBA Draft Prediction - Data Exploration Script
===============================================
This script explores all data files and generates a comprehensive report
of the dataset structure, statistics, and quality checks.

Author: Sam
Date: December 2025
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "RAW"

# Files to explore
FILES = {
    "college_2021": "CollegeBasketballPlayers2009-2021.csv",
    "college_2022": "CollegeBasketballPlayers2022.csv",
    "drafted_players": "DraftedPlayers2009-2021.xlsx",
    "latest_raptor": "latest_RAPTOR_by_player.csv",
    "modern_raptor": "modern_RAPTOR_by_player.csv",
}


# ============================================================================
# Helper Functions
# ============================================================================


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")


def explore_dataframe(df, name):
    """Comprehensive exploration of a single dataframe"""
    print_subsection(f"Dataset: {name}")

    # Basic info
    print(f"\n📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Column info
    print(f"\n📋 Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = (df[col].isna().sum() / len(df)) * 100
        print(
            f"  {i:2d}. {col:40s} | {str(dtype):12s} | {non_null:6,} non-null ({null_pct:.1f}% missing)"
        )

    # Data types summary
    print(f"\n📊 Data Types Summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    # Missing values
    print(f"\n⚠️  Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Count": missing, "Missing %": missing_pct}
    ).sort_values("Missing Count", ascending=False)

    if missing.sum() > 0:
        print(missing_df[missing_df["Missing Count"] > 0].head(10).to_string())
    else:
        print("  ✅ No missing values!")

    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Numeric Columns Summary:")
        print(df[numeric_cols].describe().round(2).to_string())

    # Sample rows
    print(f"\n🔍 First 3 Rows:")
    print(df.head(3).to_string())

    # Unique values for object columns
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0 and len(object_cols) <= 10:
        print(f"\n🏷️  Categorical Columns - Unique Values:")
        for col in object_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count <= 20:
                print(f"    → {df[col].value_counts().head(10).to_dict()}")

    return df


def check_year_ranges(df, date_col=None):
    """Check year ranges in the data"""
    year_cols = [
        col for col in df.columns if "year" in col.lower() or "season" in col.lower()
    ]

    if date_col and date_col in df.columns:
        year_cols.append(date_col)

    if year_cols:
        print(f"\n📅 Year/Season Information:")
        for col in year_cols:
            if df[col].dtype in ["int64", "float64"]:
                print(f"  {col}: {df[col].min():.0f} to {df[col].max():.0f}")
            else:
                print(f"  {col}: {df[col].nunique()} unique values")


def check_duplicates(df, subset=None):
    """Check for duplicate rows"""
    if subset:
        dup_count = df.duplicated(subset=subset).sum()
        print(f"\n🔄 Duplicates (on {subset}): {dup_count:,} rows")
    else:
        dup_count = df.duplicated().sum()
        print(f"\n🔄 Duplicate Rows: {dup_count:,}")


# ============================================================================
# Main Exploration
# ============================================================================


def main():
    print_section("NBA DRAFT PREDICTION PROJECT - DATA EXPLORATION")
    print(f"\n📁 Data Directory: {DATA_DIR}")
    print(f"🕐 Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if directory exists
    if not DATA_DIR.exists():
        print(f"\n❌ ERROR: Directory not found: {DATA_DIR}")
        return

    # Dictionary to store loaded dataframes
    dfs = {}

    # ========================================================================
    # 1. College Basketball Players Data (2009-2021)
    # ========================================================================
    print_section("1. COLLEGE BASKETBALL PLAYERS (2009-2021)")

    try:
        file_path = DATA_DIR / FILES["college_2021"]
        print(f"📂 Loading: {file_path}")
        df_college = pd.read_csv(file_path)
        dfs["college_2021"] = explore_dataframe(df_college, "College Players 2009-2021")
        check_year_ranges(df_college)
        check_duplicates(
            df_college,
            subset=(
                ["player_name", "year"] if "player_name" in df_college.columns else None
            ),
        )
    except Exception as e:
        print(f"❌ Error loading college 2021 data: {e}")

    # ========================================================================
    # 2. College Basketball Players Data (2022)
    # ========================================================================
    print_section("2. COLLEGE BASKETBALL PLAYERS (2022)")

    try:
        file_path = DATA_DIR / FILES["college_2022"]
        print(f"📂 Loading: {file_path}")
        df_college_2022 = pd.read_csv(file_path)
        dfs["college_2022"] = explore_dataframe(df_college_2022, "College Players 2022")
        check_year_ranges(df_college_2022)
    except Exception as e:
        print(f"❌ Error loading college 2022 data: {e}")

    # ========================================================================
    # 3. Drafted Players Data
    # ========================================================================
    print_section("3. DRAFTED PLAYERS (2009-2021)")

    try:
        file_path = DATA_DIR / FILES["drafted_players"]
        print(f"📂 Loading: {file_path}")
        df_drafted = pd.read_excel(file_path)
        dfs["drafted"] = explore_dataframe(df_drafted, "Drafted Players 2009-2021")
        check_year_ranges(df_drafted)

        # Draft pick distribution
        if (
            "Pk" in df_drafted.columns
            or "pick" in df_drafted.columns.str.lower().tolist()
        ):
            pick_col = (
                "Pk"
                if "Pk" in df_drafted.columns
                else [c for c in df_drafted.columns if "pick" in c.lower()][0]
            )
            print(f"\n🎯 Draft Pick Distribution:")
            print(f"  Min Pick: {df_drafted[pick_col].min()}")
            print(f"  Max Pick: {df_drafted[pick_col].max()}")
            print(f"  Avg Pick: {df_drafted[pick_col].mean():.1f}")
            print(f"  Median Pick: {df_drafted[pick_col].median():.1f}")
    except Exception as e:
        print(f"❌ Error loading drafted players data: {e}")

    # ========================================================================
    # 4. Latest RAPTOR Data
    # ========================================================================
    print_section("4. LATEST RAPTOR (NBA Performance Metrics)")

    try:
        file_path = DATA_DIR / FILES["latest_raptor"]
        print(f"📂 Loading: {file_path}")
        df_raptor_latest = pd.read_csv(file_path)
        dfs["raptor_latest"] = explore_dataframe(df_raptor_latest, "Latest RAPTOR")
        check_year_ranges(df_raptor_latest)
    except Exception as e:
        print(f"❌ Error loading latest RAPTOR data: {e}")

    # ========================================================================
    # 5. Modern RAPTOR Data
    # ========================================================================
    print_section("5. MODERN RAPTOR (NBA Performance Metrics)")

    try:
        file_path = DATA_DIR / FILES["modern_raptor"]
        print(f"📂 Loading: {file_path}")
        df_raptor_modern = pd.read_csv(file_path)
        dfs["raptor_modern"] = explore_dataframe(df_raptor_modern, "Modern RAPTOR")
        check_year_ranges(df_raptor_modern)
    except Exception as e:
        print(f"❌ Error loading modern RAPTOR data: {e}")

    # ========================================================================
    # Data Integration Analysis
    # ========================================================================
    print_section("6. DATA INTEGRATION OPPORTUNITIES")

    print("\n🔗 Potential Join Keys:")

    # Check common columns across datasets
    if "college_2021" in dfs and "drafted" in dfs:
        college_cols = set(dfs["college_2021"].columns)
        drafted_cols = set(dfs["drafted"].columns)
        common = college_cols.intersection(drafted_cols)
        print(f"\n  College ↔ Drafted: {len(common)} common columns")
        if common:
            print(f"    → {', '.join(list(common)[:10])}")

    if "drafted" in dfs and "raptor_latest" in dfs:
        drafted_cols = set(dfs["drafted"].columns)
        raptor_cols = set(dfs["raptor_latest"].columns)
        common = drafted_cols.intersection(raptor_cols)
        print(f"\n  Drafted ↔ RAPTOR: {len(common)} common columns")
        if common:
            print(f"    → {', '.join(list(common)[:10])}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n✅ Data Files Loaded:")
    for name, df in dfs.items():
        print(f"  • {name:20s}: {df.shape[0]:6,} rows × {df.shape[1]:2d} cols")

    print("\n" + "=" * 80)
    print("🎉 Data exploration complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
