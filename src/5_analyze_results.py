"""
Model Results Analyzer
======================
Explore and analyze the trained model results.
"""

import pandas as pd
from pathlib import Path

# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "Models"

print("=" * 80)
print("MODEL RESULTS ANALYZER")
print("=" * 80)

# ============================================================================
# 1. Load Results
# ============================================================================
print("\n1️⃣ Loading model results...")

try:
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    clf_importance = pd.read_csv(OUTPUT_DIR / "classifier_feature_importance.csv")
    reg_importance = pd.read_csv(OUTPUT_DIR / "regressor_feature_importance.csv")

    print(f"   ✅ Predictions: {len(predictions):,} players")
    print(f"   ✅ Classifier features: {len(clf_importance)}")
    print(f"   ✅ Regressor features: {len(reg_importance)}")
except FileNotFoundError as e:
    print(f"   ❌ Error: {e}")
    print(f"   Run model_training.py first!")
    exit(1)

# ============================================================================
# 2. Classifier Analysis
# ============================================================================
print("\n" + "=" * 80)
print("2️⃣ CLASSIFIER ANALYSIS")
print("=" * 80)

print("\n   Top 10 Most Important Features for Draft Prediction:")
print(clf_importance.head(10).to_string(index=False))

# Check prediction distribution
print(f"\n   Prediction Distribution:")
print(f"   ├─ Predicted Drafted: {predictions['predicted_drafted'].sum():,}")
print(f"   ├─ Predicted Undrafted: {(predictions['predicted_drafted'] == 0).sum():,}")
print(f"   └─ Actually Drafted: {predictions['was_drafted'].sum():,}")

# High confidence predictions
high_conf_drafted = predictions[predictions["predicted_drafted_proba"] > 0.8]
print(f"\n   High Confidence Predictions (>80%):")
print(f"   └─ Predicted to be drafted: {len(high_conf_drafted):,} players")

if len(high_conf_drafted) > 0:
    print(f"\n   Top 10 High Confidence Draft Predictions:")
    top_preds = high_conf_drafted.nlargest(10, "predicted_drafted_proba")
    print(
        top_preds[
            [
                "player_name_clean",
                "year",
                "was_drafted",
                "draft_pick",
                "predicted_drafted_proba",
                "ppg",
                "bpm",
            ]
        ].to_string(index=False)
    )

# ============================================================================
# 3. Regressor Analysis
# ============================================================================
print("\n" + "=" * 80)
print("3️⃣ REGRESSOR ANALYSIS")
print("=" * 80)

print("\n   Top 10 Most Important Features for Draft Pick Prediction:")
print(reg_importance.head(10).to_string(index=False))

# Analyze pick predictions
drafted_with_pred = predictions[predictions["predicted_pick"].notna()]

if len(drafted_with_pred) > 0:
    print(f"\n   Draft Pick Prediction Analysis:")
    print(f"   ├─ Players with pick predictions: {len(drafted_with_pred):,}")
    print(
        f"   ├─ Average predicted pick: {drafted_with_pred['predicted_pick'].mean():.1f}"
    )
    print(f"   └─ Average actual pick: {drafted_with_pred['draft_pick'].mean():.1f}")

    # Show biggest prediction errors
    drafted_with_pred["pick_error"] = abs(
        drafted_with_pred["draft_pick"] - drafted_with_pred["predicted_pick"]
    )

    print(f"\n   Biggest Prediction Errors (model was way off):")
    biggest_errors = drafted_with_pred.nlargest(5, "pick_error")
    print(
        biggest_errors[
            [
                "player_name_clean",
                "year",
                "draft_pick",
                "predicted_pick",
                "pick_error",
                "ppg",
                "bpm",
            ]
        ].to_string(index=False)
    )

# ============================================================================
# 4. Diamond Players Analysis
# ============================================================================
print("\n" + "=" * 80)
print("4️⃣ DIAMOND PLAYERS ANALYSIS 💎")
print("=" * 80)

# Actual diamonds
actual_diamonds = predictions[predictions["is_diamond"] == True]

if len(actual_diamonds) > 0:
    print(f"\n   Real Diamond Players: {len(actual_diamonds):,}")
    print(f"\n   Top 10 Diamonds (Best NBA Performance):")
    top_diamonds = actual_diamonds.nlargest(10, "raptor_total_mean")
    print(
        top_diamonds[
            [
                "player_name_clean",
                "year",
                "draft_pick",
                "ppg",
                "bpm",
                "raptor_total_mean",
                "predicted_drafted_proba",
            ]
        ].to_string(index=False)
    )

    # Model detection rate
    if "is_model_diamond" in predictions.columns:
        detected = (predictions["is_diamond"] & predictions["is_model_diamond"]).sum()
        detection_rate = detected / len(actual_diamonds) * 100
        print(
            f"\n   Model Diamond Detection Rate: {detection_rate:.1f}% ({detected}/{len(actual_diamonds)})"
        )

# ============================================================================
# 5. Interesting Cases
# ============================================================================
print("\n" + "=" * 80)
print("5️⃣ INTERESTING CASES")
print("=" * 80)

# Case 1: High confidence prediction but wasn't drafted
false_positives = predictions[
    (predictions["predicted_drafted_proba"] > 0.7) & (predictions["was_drafted"] == 0)
]

if len(false_positives) > 0:
    print(
        f"\n   🤔 Model Predicted Drafted (High Conf) but Wasn't: {len(false_positives)}"
    )
    print(f"   (These might be overlooked talents!)")
    print(
        false_positives.nlargest(5, "predicted_drafted_proba")[
            [
                "player_name_clean",
                "year",
                "predicted_drafted_proba",
                "ppg",
                "bpm",
                "raptor_total_mean",
            ]
        ].to_string(index=False)
    )

# Case 2: Low confidence but got drafted anyway
surprises = predictions[
    (predictions["predicted_drafted_proba"] < 0.3) & (predictions["was_drafted"] == 1)
]

if len(surprises) > 0:
    print(f"\n   😲 Drafted Despite Low Model Confidence: {len(surprises)}")
    print(f"   (Model missed these - what made them draftable?)")
    print(
        surprises[
            [
                "player_name_clean",
                "year",
                "draft_pick",
                "predicted_drafted_proba",
                "ppg",
                "bpm",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

# Case 3: Undrafted but had NBA success
undrafted_nba_success = predictions[
    (predictions["was_drafted"] == 0)
    & (predictions["raptor_total_mean"] > 0)
    & predictions["raptor_total_mean"].notna()
]

if len(undrafted_nba_success) > 0:
    print(f"\n   🌟 Undrafted Players Who Made It: {len(undrafted_nba_success)}")
    print(
        undrafted_nba_success.nlargest(5, "raptor_total_mean")[
            [
                "player_name_clean",
                "year",
                "ppg",
                "bpm",
                "raptor_total_mean",
                "predicted_drafted_proba",
            ]
        ].to_string(index=False)
    )

# ============================================================================
# 6. Feature Comparison
# ============================================================================
print("\n" + "=" * 80)
print("6️⃣ FEATURE IMPORTANCE COMPARISON")
print("=" * 80)

# Compare top features between classifier and regressor
print("\n   Features important for BOTH models:")

clf_top = set(clf_importance.head(20)["feature"])
reg_top = set(reg_importance.head(20)["feature"])
common = clf_top & reg_top

if common:
    print(f"   Found {len(common)} common features in top 20:")
    for feat in list(common)[:10]:
        clf_rank = clf_importance[clf_importance["feature"] == feat].index[0] + 1
        reg_rank = reg_importance[reg_importance["feature"] == feat].index[0] + 1
        print(f"   - {feat:30s} (CLF rank: {clf_rank:2d}, REG rank: {reg_rank:2d})")

# Features unique to classifier
clf_unique = clf_top - reg_top
if clf_unique:
    print(f"\n   Unique to CLASSIFIER (draft status):")
    for feat in list(clf_unique)[:5]:
        print(f"   - {feat}")

# Features unique to regressor
reg_unique = reg_top - clf_top
if reg_unique:
    print(f"\n   Unique to REGRESSOR (draft pick):")
    for feat in list(reg_unique)[:5]:
        print(f"   - {feat}")

# ============================================================================
# 7. Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"""
   📊 Key Insights:
   ├─ Total test players: {len(predictions):,}
   ├─ Drafted: {predictions['was_drafted'].sum():,}
   ├─ Diamond players: {len(actual_diamonds):,}
   └─ High-confidence predictions: {(predictions['predicted_drafted_proba'] > 0.8).sum():,}
   
   💡 Interesting Findings:
   ├─ False positives (model said yes, reality no): {len(false_positives):,}
   ├─ Surprises (drafted despite low conf): {len(surprises):,}
   └─ Undrafted NBA success stories: {len(undrafted_nba_success):,}
   
   📁 Full results in: {OUTPUT_DIR}
"""
)
