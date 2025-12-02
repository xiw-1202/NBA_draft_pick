"""
Model Training - NBA Draft Prediction with LightGBM
====================================================
Train classifier and regressor models to predict draft outcomes.

Author: Sam
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "PROCESSED"
OUTPUT_DIR = PROJECT_ROOT / "Models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MODEL TRAINING - NBA DRAFT PREDICTION")
print("=" * 80)

# ============================================================================
# 1. Load Feature-Engineered Data
# ============================================================================
print("\n1️⃣ Loading feature-engineered data...")

df = pd.read_csv(DATA_DIR / "featured_data.csv")
features = pd.read_csv(DATA_DIR / "feature_names.csv")["feature"].tolist()

print(f"   Dataset: {df.shape}")
print(f"   Features: {len(features)}")
print(f"   Players: {len(df):,}")

# ============================================================================
# 2. Time-Based Train/Test Split
# ============================================================================
print("\n2️⃣ Creating time-based train/test split...")

# Train: 2009-2018, Test: 2019-2021
train_df = df[df["year"] <= 2018].copy()
test_df = df[df["year"] >= 2019].copy()

print(
    f"   Train (2009-2018): {len(train_df):,} players ({train_df['year'].min()}-{train_df['year'].max()})"
)
print(
    f"   Test (2019-2021):  {len(test_df):,} players ({test_df['year'].min()}-{test_df['year'].max()})"
)

print(
    f"\n   Train drafted: {train_df['was_drafted'].sum():,} ({train_df['was_drafted'].mean()*100:.2f}%)"
)
print(
    f"   Test drafted:  {test_df['was_drafted'].sum():,} ({test_df['was_drafted'].mean()*100:.2f}%)"
)

# ============================================================================
# 3. Prepare Data for Classification (Drafted vs Undrafted)
# ============================================================================
print("\n" + "=" * 80)
print("3️⃣ CLASSIFIER: Predicting Draft Status")
print("=" * 80)

# Features and targets
X_train = train_df[features].fillna(0)
y_train = train_df["was_drafted"].astype(int)

X_test = test_df[features].fillna(0)
y_test = test_df["was_drafted"].astype(int)

print(f"\n   Train: X={X_train.shape}, y={y_train.sum():,} drafted")
print(f"   Test:  X={X_test.shape}, y={y_test.sum():,} drafted")

# Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   Class imbalance ratio: {scale_pos_weight:.2f}:1 (undrafted:drafted)")

# ============================================================================
# 4. Train LightGBM Classifier
# ============================================================================
print("\n   Training LightGBM Classifier...")

clf_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": scale_pos_weight,
    "verbose": -1,
    "random_state": 42,
    "min_child_samples": 20,
}

# Create datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train with validation
clf_model = lgb.train(
    clf_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

print(f"   ✅ Training complete! Best iteration: {clf_model.best_iteration}")

# ============================================================================
# 5. Evaluate Classifier
# ============================================================================
print("\n   📊 Evaluating Classifier...")

# Predictions
y_pred_proba = clf_model.predict(X_test, num_iteration=clf_model.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n   Classifier Performance:")
print(f"   ├─ Accuracy:  {accuracy:.4f}")
print(f"   ├─ Precision: {precision:.4f}")
print(f"   ├─ Recall:    {recall:.4f}")
print(f"   ├─ F1-Score:  {f1:.4f}")
print(f"   └─ ROC-AUC:   {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n   Confusion Matrix:")
print(f"   ┌─────────────┬──────────┬──────────┐")
print(f"   │             │ Pred: No │ Pred: Yes│")
print(f"   ├─────────────┼──────────┼──────────┤")
print(f"   │ Actual: No  │ {cm[0,0]:8,} │ {cm[0,1]:8,} │  TN / FP")
print(f"   │ Actual: Yes │ {cm[1,0]:8,} │ {cm[1,1]:8,} │  FN / TP")
print(f"   └─────────────┴──────────┴──────────┘")

# Classification report
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Undrafted", "Drafted"]))

# ============================================================================
# 6. Feature Importance (Classifier)
# ============================================================================
print("\n   📋 Feature Importance (Classifier)...")

feature_importance = pd.DataFrame(
    {
        "feature": features,
        "importance": clf_model.feature_importance(importance_type="gain"),
    }
).sort_values("importance", ascending=False)

print(f"\n   Top 20 Features for Draft Prediction:")
print(feature_importance.head(20).to_string(index=False))

# Save feature importance
feature_importance.to_csv(OUTPUT_DIR / "classifier_feature_importance.csv", index=False)

# ============================================================================
# 7. Save Classifier
# ============================================================================
print("\n   💾 Saving classifier...")

clf_model.save_model(str(OUTPUT_DIR / "lightgbm_classifier.txt"))
print(f"   ✅ Saved to: {OUTPUT_DIR / 'lightgbm_classifier.txt'}")

# ============================================================================
# 8. Prepare Data for Regression (Draft Pick Position)
# ============================================================================
print("\n" + "=" * 80)
print("8️⃣ REGRESSOR: Predicting Draft Pick Position")
print("=" * 80)

# Filter to only drafted players
train_drafted = train_df[train_df["was_drafted"] == 1].copy()
test_drafted = test_df[test_df["was_drafted"] == 1].copy()

X_train_reg = train_drafted[features].fillna(0)
y_train_reg = train_drafted["draft_pick"].fillna(60)

X_test_reg = test_drafted[features].fillna(0)
y_test_reg = test_drafted["draft_pick"].fillna(60)

print(f"\n   Train (drafted only): {len(X_train_reg):,} players")
print(f"   Test (drafted only):  {len(X_test_reg):,} players")
print(f"   Pick range - Train: {y_train_reg.min():.0f} to {y_train_reg.max():.0f}")
print(f"   Pick range - Test:  {y_test_reg.min():.0f} to {y_test_reg.max():.0f}")

# ============================================================================
# 9. Train LightGBM Regressor
# ============================================================================
print("\n   Training LightGBM Regressor...")

reg_params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
    "min_child_samples": 10,
}

# Create datasets
train_data_reg = lgb.Dataset(X_train_reg, label=y_train_reg)
test_data_reg = lgb.Dataset(X_test_reg, label=y_test_reg, reference=train_data_reg)

# Train with validation
reg_model = lgb.train(
    reg_params,
    train_data_reg,
    num_boost_round=1000,
    valid_sets=[train_data_reg, test_data_reg],
    valid_names=["train", "test"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

print(f"   ✅ Training complete! Best iteration: {reg_model.best_iteration}")

# ============================================================================
# 10. Evaluate Regressor
# ============================================================================
print("\n   📊 Evaluating Regressor...")

# Predictions
y_pred_reg = reg_model.predict(X_test_reg, num_iteration=reg_model.best_iteration)
y_pred_reg_clipped = np.clip(y_pred_reg, 1, 60)

# Metrics
mae = mean_absolute_error(y_test_reg, y_pred_reg_clipped)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg_clipped))
r2 = r2_score(y_test_reg, y_pred_reg_clipped)

print(f"\n   Regressor Performance:")
print(f"   ├─ MAE (Mean Absolute Error): {mae:.2f} picks")
print(f"   ├─ RMSE (Root Mean Squared Error): {rmse:.2f} picks")
print(f"   └─ R² Score: {r2:.4f}")

# Accuracy within X picks
within_5 = (np.abs(y_test_reg - y_pred_reg_clipped) <= 5).mean()
within_10 = (np.abs(y_test_reg - y_pred_reg_clipped) <= 10).mean()
within_15 = (np.abs(y_test_reg - y_pred_reg_clipped) <= 15).mean()

print(f"\n   Prediction Accuracy:")
print(f"   ├─ Within  5 picks: {within_5*100:.1f}%")
print(f"   ├─ Within 10 picks: {within_10*100:.1f}%")
print(f"   └─ Within 15 picks: {within_15*100:.1f}%")

# ============================================================================
# 11. Feature Importance (Regressor)
# ============================================================================
print("\n   📋 Feature Importance (Regressor)...")

feature_importance_reg = pd.DataFrame(
    {
        "feature": features,
        "importance": reg_model.feature_importance(importance_type="gain"),
    }
).sort_values("importance", ascending=False)

print(f"\n   Top 20 Features for Draft Pick Prediction:")
print(feature_importance_reg.head(20).to_string(index=False))

# Save feature importance
feature_importance_reg.to_csv(
    OUTPUT_DIR / "regressor_feature_importance.csv", index=False
)

# ============================================================================
# 12. Save Regressor
# ============================================================================
print("\n   💾 Saving regressor...")

reg_model.save_model(str(OUTPUT_DIR / "lightgbm_regressor.txt"))
print(f"   ✅ Saved to: {OUTPUT_DIR / 'lightgbm_regressor.txt'}")

# ============================================================================
# 13. Diamond Player Detection
# ============================================================================
print("\n" + "=" * 80)
print("1️⃣3️⃣ DIAMOND PLAYER DETECTION")
print("=" * 80)

# Add predictions to test set
test_results = test_df.copy()
test_results["predicted_drafted_proba"] = clf_model.predict(
    test_df[features].fillna(0), num_iteration=clf_model.best_iteration
)
test_results["predicted_drafted"] = (
    test_results["predicted_drafted_proba"] >= 0.5
).astype(int)

# For drafted players, predict their pick
drafted_mask = test_results["was_drafted"] == 1
if drafted_mask.sum() > 0:
    test_results.loc[drafted_mask, "predicted_pick"] = reg_model.predict(
        test_results.loc[drafted_mask, features].fillna(0),
        num_iteration=reg_model.best_iteration,
    )
    test_results.loc[drafted_mask, "predicted_pick"] = test_results.loc[
        drafted_mask, "predicted_pick"
    ].clip(1, 60)

# Diamond detection strategy:
# 1. Model predicted low draft probability OR predicted late pick
# 2. But player has high NBA performance (RAPTOR > 0)

test_results["model_predicted_low"] = (
    test_results["predicted_drafted_proba"] < 0.3
) | (  # Low draft probability
    test_results["predicted_pick"] > 30
)  # Or predicted 2nd round

test_results["is_model_diamond"] = test_results["model_predicted_low"] & (
    test_results["raptor_total_mean"] > 0
)

# Actual diamonds (from ground truth)
actual_diamonds = test_results[test_results["is_diamond"] == True]
predicted_diamonds = test_results[test_results["is_model_diamond"] == True]

print(f"\n   💎 Diamond Player Analysis:")
print(f"   ├─ Actual diamonds in test set: {len(actual_diamonds):,}")
print(f"   ├─ Model predicted diamonds: {len(predicted_diamonds):,}")
print(
    f"   └─ Overlap (correctly identified): {(test_results['is_diamond'] & test_results['is_model_diamond']).sum():,}"
)

# Show actual diamonds
if len(actual_diamonds) > 0:
    print(f"\n   🏆 Top 10 Real Diamond Players (by NBA RAPTOR):")
    diamond_cols = [
        "player_name_clean",
        "year",
        "draft_pick",
        "ppg",
        "bpm",
        "raptor_total_mean",
        "predicted_drafted_proba",
    ]
    top_diamonds = actual_diamonds.nlargest(10, "raptor_total_mean")[diamond_cols]
    print(top_diamonds.to_string(index=False))

# Show model's diamond predictions that were actually drafted high (misses)
model_diamonds_drafted_high = test_results[
    test_results["is_model_diamond"]
    & (test_results["draft_pick"] <= 30)
    & test_results["draft_pick"].notna()
]

if len(model_diamonds_drafted_high) > 0:
    print(
        f"\n   ⚠️  Model Predicted Diamonds but Drafted High (False Alarms): {len(model_diamonds_drafted_high)}"
    )
    print(model_diamonds_drafted_high[diamond_cols].head(5).to_string(index=False))

# ============================================================================
# 14. Save Predictions and Results
# ============================================================================
print("\n1️⃣4️⃣ Saving predictions and results...")

# Save test predictions
prediction_cols = [
    "player_name_clean",
    "year",
    "team",
    "conf",
    "yr",
    "was_drafted",
    "draft_pick",
    "ppg",
    "rpg",
    "apg",
    "bpm",
    "predicted_drafted",
    "predicted_drafted_proba",
    "predicted_pick",
    "raptor_total_mean",
    "war_total_sum",
    "is_diamond",
    "is_model_diamond",
]

test_predictions = test_results[
    [col for col in prediction_cols if col in test_results.columns]
]
test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
print(f"   ✅ Test predictions saved to: test_predictions.csv")

# Save actual diamonds
if len(actual_diamonds) > 0:
    actual_diamonds.to_csv(OUTPUT_DIR / "actual_diamond_players.csv", index=False)
    print(
        f"   ✅ Actual diamonds saved to: actual_diamond_players.csv ({len(actual_diamonds)} players)"
    )

# Save predicted diamonds
if len(predicted_diamonds) > 0:
    predicted_diamonds.to_csv(OUTPUT_DIR / "predicted_diamond_players.csv", index=False)
    print(
        f"   ✅ Predicted diamonds saved to: predicted_diamond_players.csv ({len(predicted_diamonds)} players)"
    )

# ============================================================================
# 15. Create Summary Report
# ============================================================================
print("\n1️⃣5️⃣ Creating summary report...")

summary_report = f"""
================================================================================
NBA DRAFT PREDICTION - MODEL TRAINING SUMMARY
================================================================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET:
--------
Training Set (2009-2018): {len(train_df):,} players
Test Set (2019-2021):     {len(test_df):,} players
Features Used:            {len(features)}

CLASSIFIER PERFORMANCE (Draft Status):
--------------------------------------
Accuracy:   {accuracy:.4f}
Precision:  {precision:.4f}
Recall:     {recall:.4f}
F1-Score:   {f1:.4f}
ROC-AUC:    {roc_auc:.4f}

Confusion Matrix:
                Predicted
                No      Yes
Actual  No      {cm[0,0]:,}   {cm[0,1]:,}
        Yes     {cm[1,0]:,}     {cm[1,1]:,}

REGRESSOR PERFORMANCE (Draft Pick):
------------------------------------
MAE (Mean Absolute Error):  {mae:.2f} picks
RMSE:                       {rmse:.2f} picks
R² Score:                   {r2:.4f}

Prediction Accuracy:
- Within  5 picks: {within_5*100:.1f}%
- Within 10 picks: {within_10*100:.1f}%
- Within 15 picks: {within_15*100:.1f}%

DIAMOND PLAYER DETECTION:
--------------------------
Actual diamonds in test:      {len(actual_diamonds):,}
Model predicted diamonds:     {len(predicted_diamonds):,}
Correctly identified:         {(test_results['is_diamond'] & test_results['is_model_diamond']).sum():,}

TOP FEATURES (CLASSIFIER):
--------------------------
{feature_importance.head(10).to_string(index=False)}

TOP FEATURES (REGRESSOR):
-------------------------
{feature_importance_reg.head(10).to_string(index=False)}

OUTPUT FILES:
-------------
✅ Models/lightgbm_classifier.txt
✅ Models/lightgbm_regressor.txt
✅ Models/classifier_feature_importance.csv
✅ Models/regressor_feature_importance.csv
✅ Models/test_predictions.csv
✅ Models/actual_diamond_players.csv
✅ Models/predicted_diamond_players.csv
✅ Models/training_summary.txt

================================================================================
"""

summary_file = OUTPUT_DIR / "training_summary.txt"
with open(summary_file, "w") as f:
    f.write(summary_report)

print(summary_report)

print("=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📁 All results saved to: {OUTPUT_DIR}")
print(f"\n🎯 Next steps:")
print(f"   1. Review feature importance: head Models/classifier_feature_importance.csv")
print(f"   2. Check diamond players: head Models/actual_diamond_players.csv")
print(f"   3. Analyze predictions: head Models/test_predictions.csv")
print(f"   4. Create visualizations (optional)")
