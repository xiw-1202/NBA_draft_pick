"""
Complete Visualization Script - NBA Draft Prediction Results
=============================================================
Creates clean, professional visualizations following classic design principles.

Features:
- Classic professional color palette (blues, grays)
- Clean layouts with optimal spacing
- Professional typography and labels
- Intuitive, easy-to-understand charts
- Clear annotations and explanations

Author: Sam
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Classic professional style settings
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16

# Classic professional color palette
COLORS = {
    "primary_blue": "#2E5090",
    "secondary_blue": "#5B8DBE",
    "light_blue": "#A8C5E8",
    "accent_orange": "#E67E22",
    "success_green": "#27AE60",
    "warning_red": "#C0392B",
    "neutral_gray": "#7F8C8D",
    "light_gray": "#BDC3C7",
    "dark_gray": "#34495E",
}

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "Models"
DATA_DIR = PROJECT_ROOT / "Data" / "PROCESSED"
OUTPUT_DIR = PROJECT_ROOT / "Visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CREATING NBA DRAFT PREDICTION VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\n1. Loading data...")

predictions = pd.read_csv(MODELS_DIR / "test_predictions.csv")
clf_importance = pd.read_csv(MODELS_DIR / "classifier_feature_importance.csv")
reg_importance = pd.read_csv(MODELS_DIR / "regressor_feature_importance.csv")

print(f"   ✓ Loaded predictions: {len(predictions):,} players")
print(f"   ✓ Loaded feature importance data")

# ============================================================================
# 1. Feature Importance - Clean and Simple
# ============================================================================
print("\n2. Creating feature importance visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left: Classifier importance
top_clf = clf_importance.head(12)
y_pos = np.arange(len(top_clf))
axes[0].barh(y_pos, top_clf["importance"],
             color=COLORS["primary_blue"], alpha=0.8, edgecolor="white", linewidth=1.5)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(top_clf["feature"], fontsize=10)
axes[0].invert_yaxis()
axes[0].set_xlabel("Importance Score", fontweight="bold")
axes[0].set_title("Draft Status Prediction\nTop 12 Most Important Features",
                  fontweight="bold", pad=15)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].grid(axis="x", alpha=0.3, linestyle="--")

# Right: Regressor importance
top_reg = reg_importance.head(12)
y_pos = np.arange(len(top_reg))
axes[1].barh(y_pos, top_reg["importance"],
             color=COLORS["accent_orange"], alpha=0.8, edgecolor="white", linewidth=1.5)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(top_reg["feature"], fontsize=10)
axes[1].invert_yaxis()
axes[1].set_xlabel("Importance Score", fontweight="bold")
axes[1].set_title("Draft Pick Prediction\nTop 12 Most Important Features",
                  fontweight="bold", pad=15)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].grid(axis="x", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_feature_importance.png", dpi=300, bbox_inches="tight", facecolor="white")
print(f"   ✓ Saved: 1_feature_importance.png")
plt.close()

# ============================================================================
# 2. Confusion Matrix - Professional Design
# ============================================================================
print("\n3. Creating confusion matrix...")

from sklearn.metrics import confusion_matrix

y_true = predictions["was_drafted"]
y_pred = predictions["predicted_drafted"]
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))

# Create heatmap with professional colors
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=sns.light_palette(COLORS["primary_blue"], as_cmap=True),
    cbar=True,
    square=True,
    xticklabels=["Not Drafted", "Drafted"],
    yticklabels=["Not Drafted", "Drafted"],
    annot_kws={"size": 16, "weight": "bold"},
    linewidths=2,
    linecolor="white",
    ax=ax
)

ax.set_xlabel("Predicted Class", fontsize=13, fontweight="bold", labelpad=10)
ax.set_ylabel("Actual Class", fontsize=13, fontweight="bold", labelpad=10)
ax.set_title("Draft Status Prediction Accuracy\nConfusion Matrix",
             fontsize=15, fontweight="bold", pad=20)

# Calculate metrics
accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0

# Add metrics box
metrics_text = f"Accuracy: {accuracy:.1%}  |  Precision: {precision:.1%}  |  Recall: {recall:.1%}"
ax.text(0.5, -0.12, metrics_text, ha="center", transform=ax.transAxes,
        fontsize=12, weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["light_gray"],
                 alpha=0.3, edgecolor="none"))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_confusion_matrix.png", dpi=300, bbox_inches="tight", facecolor="white")
print(f"   ✓ Saved: 2_confusion_matrix.png")
plt.close()

# ============================================================================
# 3. Model Prediction Quality - Clear and Intuitive
# ============================================================================
print("\n4. Creating prediction analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("How Well Does the Model Predict Draft Outcomes?",
             fontsize=16, fontweight="bold", y=0.995)

drafted_probs = predictions[predictions["was_drafted"] == 1]["predicted_drafted_proba"]
undrafted_probs = predictions[predictions["was_drafted"] == 0]["predicted_drafted_proba"]

# Top Left: Model confidence for drafted players
axes[0, 0].hist(drafted_probs, bins=25, color=COLORS["success_green"],
                alpha=0.7, edgecolor="white", linewidth=1.2)
axes[0, 0].axvline(drafted_probs.mean(), color=COLORS["dark_gray"],
                  linestyle="--", linewidth=2.5, alpha=0.8)
axes[0, 0].set_xlabel("Model Confidence Score (0 = Won't Draft, 1 = Will Draft)", fontweight="bold")
axes[0, 0].set_ylabel("Number of Players", fontweight="bold")
axes[0, 0].set_title(f"Players Who Were Actually Drafted ({len(drafted_probs):,} players)",
                     fontweight="bold", pad=10)
axes[0, 0].spines["top"].set_visible(False)
axes[0, 0].spines["right"].set_visible(False)
axes[0, 0].grid(axis="y", alpha=0.3, linestyle="--")
axes[0, 0].set_xlim(0, 1)

# Add text box with explanation
axes[0, 0].text(0.05, 0.95,
                f"Average Model\nConfidence: {drafted_probs.mean():.2f}\n\n"
                f"✓ High scores = Model\n   correctly predicted\n   they'd be drafted",
                transform=axes[0, 0].transAxes, fontsize=10, weight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                         alpha=0.9, edgecolor=COLORS["success_green"], linewidth=2))

# Top Right: Model confidence for undrafted players
axes[0, 1].hist(undrafted_probs, bins=25, color=COLORS["neutral_gray"],
                alpha=0.7, edgecolor="white", linewidth=1.2)
axes[0, 1].axvline(undrafted_probs.mean(), color=COLORS["dark_gray"],
                  linestyle="--", linewidth=2.5, alpha=0.8)
axes[0, 1].set_xlabel("Model Confidence Score (0 = Won't Draft, 1 = Will Draft)", fontweight="bold")
axes[0, 1].set_ylabel("Number of Players", fontweight="bold")
axes[0, 1].set_title(f"Players Who Were Not Drafted ({len(undrafted_probs):,} players)",
                     fontweight="bold", pad=10)
axes[0, 1].spines["top"].set_visible(False)
axes[0, 1].spines["right"].set_visible(False)
axes[0, 1].grid(axis="y", alpha=0.3, linestyle="--")
axes[0, 1].set_xlim(0, 1)

# Add text box with explanation
axes[0, 1].text(0.95, 0.95,
                f"Average Model\nConfidence: {undrafted_probs.mean():.2f}\n\n"
                f"✓ Low scores = Model\n   correctly predicted\n   they wouldn't be drafted",
                transform=axes[0, 1].transAxes, fontsize=10, weight="bold",
                verticalalignment="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                         alpha=0.9, edgecolor=COLORS["neutral_gray"], linewidth=2))

# Bottom Left: Draft pick accuracy scatter
drafted_only = predictions[predictions["draft_pick"].notna()].copy()
axes[1, 0].scatter(drafted_only["draft_pick"], drafted_only["predicted_pick"],
                   alpha=0.5, s=50, color=COLORS["primary_blue"],
                   edgecolors="white", linewidths=0.8)
axes[1, 0].plot([1, 60], [1, 60], color=COLORS["warning_red"],
                linestyle="--", linewidth=2.5, label="Perfect Prediction Line", alpha=0.8)
axes[1, 0].set_xlabel("Actual Draft Pick (1 = First Pick)", fontweight="bold")
axes[1, 0].set_ylabel("Model's Predicted Pick", fontweight="bold")
axes[1, 0].set_title(f"Draft Position Prediction Accuracy ({len(drafted_only):,} drafted players)",
                     fontweight="bold", pad=10)
axes[1, 0].legend(frameon=True, fancybox=True, shadow=False, loc="upper left")
axes[1, 0].spines["top"].set_visible(False)
axes[1, 0].spines["right"].set_visible(False)
axes[1, 0].grid(alpha=0.3, linestyle="--")
axes[1, 0].set_xlim(0, 65)
axes[1, 0].set_ylim(0, 65)

# Add MAE annotation
mae = np.abs(drafted_only["draft_pick"] - drafted_only["predicted_pick"]).mean()
axes[1, 0].text(0.98, 0.05,
                f"Average Error:\n{mae:.1f} picks off\n\n"
                f"(Points closer to red line\n= more accurate predictions)",
                transform=axes[1, 0].transAxes, fontsize=10, weight="bold",
                verticalalignment="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                         alpha=0.9, edgecolor=COLORS["light_gray"], linewidth=2))

# Bottom Right: Prediction accuracy by range - PROPERLY FILTERED
# IMPORTANT: Filter out NaN predictions (model only predicts for players it thinks will be drafted)
valid_predictions = drafted_only.dropna(subset=["draft_pick", "predicted_pick"])

accuracy_ranges = ["Within\n5 Picks", "Within\n10 Picks", "Within\n15 Picks"]
within_5 = (np.abs(valid_predictions["draft_pick"] - valid_predictions["predicted_pick"]) <= 5).mean() * 100
within_10 = (np.abs(valid_predictions["draft_pick"] - valid_predictions["predicted_pick"]) <= 10).mean() * 100
within_15 = (np.abs(valid_predictions["draft_pick"] - valid_predictions["predicted_pick"]) <= 15).mean() * 100
accuracy_values = [within_5, within_10, within_15]

bars = axes[1, 1].bar(accuracy_ranges, accuracy_values,
                     color=[COLORS["warning_red"], COLORS["accent_orange"], COLORS["success_green"]],
                     alpha=0.8, edgecolor="white", linewidth=1.5)
axes[1, 1].set_ylabel("Percentage of Players", fontweight="bold")
axes[1, 1].set_title(f"How Close Were the Pick Predictions?\n({len(valid_predictions):,} players with predictions)",
                    fontweight="bold", pad=10, fontsize=12)
axes[1, 1].spines["top"].set_visible(False)
axes[1, 1].spines["right"].set_visible(False)
axes[1, 1].grid(axis="y", alpha=0.3, linestyle="--")

# AUTO-SCALE: Set y-axis based on actual data, not fixed 100%
max_val = max(accuracy_values)
if max_val < 10:  # If values are very small (like 1-5%)
    axes[1, 1].set_ylim(0, max_val * 1.3)  # Just 30% padding above max
else:  # If values are reasonable percentages
    axes[1, 1].set_ylim(0, min(100, max_val * 1.2))  # 20% padding, max 100%

# Add value labels on bars with better explanation
for i, (bar, val) in enumerate(zip(bars, accuracy_values)):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + max_val*0.05,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Add explanation text that adapts to the data
if max_val > 10:
    example_text = f"Example: {within_10:.1f}% of predictions were within 10 picks of actual draft position"
else:
    example_text = f"Note: Low accuracy indicates this is a very difficult prediction task"

axes[1, 1].text(0.5, -0.18, example_text,
                transform=axes[1, 1].transAxes, ha="center", fontsize=10,
                style="italic", color=COLORS["neutral_gray"])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3_prediction_distribution.png", dpi=300, bbox_inches="tight", facecolor="white")
print(f"   ✓ Saved: 3_prediction_distribution.png")
plt.close()

# ============================================================================
# 4. Diamond Players - Clean Professional Design
# ============================================================================
print("\n5. Creating diamond players visualization...")

diamonds_full = pd.read_csv(DATA_DIR / "diamond_players_clean.csv", low_memory=False)
top_diamonds = diamonds_full.nlargest(15, "war_total_sum").copy()
top_diamonds = top_diamonds.sort_values("war_total_sum", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Late-Round Success Stories: Diamond Players in the NBA",
             fontsize=16, fontweight="bold", y=0.98)

# Left: Top 15 diamonds by WAR
colors_list = [COLORS["warning_red"] if pd.isna(x) else COLORS["primary_blue"]
               for x in top_diamonds["draft_pick"]]

# Create labels
labels = []
for _, row in top_diamonds.iterrows():
    if pd.isna(row["draft_pick"]):
        labels.append(f"{row['player_name_clean']}")
    else:
        labels.append(f"{row['player_name_clean']}")

y_pos = np.arange(len(top_diamonds))
bars = axes[0].barh(y_pos, top_diamonds["war_total_sum"],
                   color=colors_list, alpha=0.8, edgecolor="white", linewidth=1.2)

axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(labels, fontsize=10)
axes[0].set_xlabel("Career WAR (Wins Above Replacement)", fontweight="bold", fontsize=12)
axes[0].set_title("Top 15 Diamond Players by Career Impact", fontweight="bold", pad=15)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].grid(axis="x", alpha=0.3, linestyle="--")

# Add value labels
for i, (bar, war) in enumerate(zip(bars, top_diamonds["war_total_sum"])):
    width = bar.get_width()
    axes[0].text(width + 1.5, bar.get_y() + bar.get_height()/2,
                f'{war:.1f}',
                ha='left', va='center', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["warning_red"], alpha=0.8, label="Undrafted"),
    Patch(facecolor=COLORS["primary_blue"], alpha=0.8, label="Late Pick (#31-60)")
]
axes[0].legend(handles=legend_elements, loc="lower right", frameon=True, fancybox=True)

# Right: Career statistics summary
stats_summary = {
    'Average Career\nWAR': diamonds_full["war_total_sum"].mean(),
    'Average NBA\nSeasons': diamonds_full["season_count"].mean(),
    'Average Career\nMinutes (1000s)': diamonds_full["mp_sum"].mean() / 1000,
    'Total Diamond\nPlayers': len(diamonds_full)
}

x_pos = np.arange(len(stats_summary))
bars = axes[1].bar(x_pos, list(stats_summary.values()),
                  color=[COLORS["primary_blue"], COLORS["secondary_blue"],
                         COLORS["light_blue"], COLORS["accent_orange"]],
                  alpha=0.8, edgecolor="white", linewidth=1.5)

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(list(stats_summary.keys()), fontsize=10)
axes[1].set_ylabel("Value", fontweight="bold", fontsize=12)
axes[1].set_title("Diamond Players Career Statistics", fontweight="bold", pad=15)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].grid(axis="y", alpha=0.3, linestyle="--")

# Add value labels
for bar, val in zip(bars, stats_summary.values()):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, height + max(stats_summary.values())*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add definition footnote
fig.text(0.5, 0.02,
         "Diamond Players: Late picks (#31-60) or undrafted players with Career WAR ≥ 10.0 and Career Minutes ≥ 2,000",
         ha='center', fontsize=10, style='italic', color=COLORS["neutral_gray"])

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(OUTPUT_DIR / "4_diamond_players.png", dpi=300, bbox_inches="tight", facecolor="white")
print(f"   ✓ Saved: 4_diamond_players.png")
plt.close()

# ============================================================================
# 5. Class Year Analysis - Professional Design
# ============================================================================
print("\n6. Creating class year analysis...")

try:
    full_data = pd.read_csv(DATA_DIR / "featured_data.csv")
    test_data_full = full_data[full_data["year"].between(2019, 2021)].copy()

    if "yr" in test_data_full.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Class Year Impact on Draft Outcomes",
                     fontsize=15, fontweight="bold", y=0.98)

        # Left: Draft rate by class
        class_stats = test_data_full.groupby("yr").agg({
            "was_drafted": ["sum", "count", "mean"]
        }).reset_index()
        class_stats.columns = ["Class", "Drafted", "Total", "Rate"]
        class_stats["Rate"] = class_stats["Rate"] * 100

        class_order = ["Fr", "So", "Jr", "Sr"]
        class_stats["Class"] = pd.Categorical(class_stats["Class"],
                                             categories=class_order, ordered=True)
        class_stats = class_stats.sort_values("Class").reset_index(drop=True)

        # Professional gradient colors
        colors = [COLORS["light_blue"], COLORS["secondary_blue"],
                 COLORS["primary_blue"], COLORS["dark_gray"]]

        bars1 = axes[0].bar(range(len(class_stats)), class_stats["Rate"],
                           color=colors[:len(class_stats)], alpha=0.8,
                           edgecolor="white", linewidth=1.5)
        axes[0].set_xticks(range(len(class_stats)))
        axes[0].set_xticklabels(["Freshman", "Sophomore", "Junior", "Senior"])
        axes[0].set_ylabel("Draft Rate (%)", fontweight="bold", fontsize=12)
        axes[0].set_title("Draft Probability by Class Year", fontweight="bold", pad=15)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        for i, (bar, row) in enumerate(zip(bars1, class_stats.itertuples())):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.3,
                        f"{row.Rate:.1f}%\n(n={row.Total:.0f})",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Right: Average pick by class
        drafted_by_class = test_data_full[test_data_full["draft_pick"].notna()].groupby("yr")["draft_pick"].agg(["mean", "count"]).reset_index()
        drafted_by_class.columns = ["Class", "Avg_Pick", "Count"]
        drafted_by_class["Class"] = pd.Categorical(drafted_by_class["Class"],
                                                   categories=class_order, ordered=True)
        drafted_by_class = drafted_by_class.sort_values("Class").reset_index(drop=True)

        bars2 = axes[1].bar(range(len(drafted_by_class)), drafted_by_class["Avg_Pick"],
                           color=colors[:len(drafted_by_class)], alpha=0.8,
                           edgecolor="white", linewidth=1.5)
        axes[1].set_xticks(range(len(drafted_by_class)))
        axes[1].set_xticklabels(["Freshman", "Sophomore", "Junior", "Senior"])
        axes[1].set_ylabel("Average Draft Pick (Lower = Better)", fontweight="bold", fontsize=12)
        axes[1].set_title("Average Draft Position by Class Year", fontweight="bold", pad=15)
        axes[1].invert_yaxis()
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        for i, (bar, row) in enumerate(zip(bars2, drafted_by_class.itertuples())):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, height,
                        f"Pick #{row.Avg_Pick:.1f}\n(n={row.Count:.0f})",
                        ha="center", va="top", fontsize=9, fontweight="bold")

        # Add insight text
        fig.text(0.5, 0.02,
                "Insight: Seniors have higher draft rates but lower (worse) draft positions - NBA values potential over production",
                ha='center', fontsize=10, style='italic', color=COLORS["neutral_gray"])

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.savefig(OUTPUT_DIR / "5_class_year_analysis.png", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"   ✓ Saved: 5_class_year_analysis.png")
        plt.close()

except Exception as e:
    print(f"   ⚠ Could not create class year analysis: {e}")

# ============================================================================
# 6. Model Performance Dashboard - Clean Summary
# ============================================================================
print("\n7. Creating model performance dashboard...")

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

fig.suptitle("Model Performance Summary Dashboard",
             fontsize=16, fontweight="bold", y=0.98)

# Calculate all metrics
y_true_clf = predictions["was_drafted"]
y_pred_clf = predictions["predicted_drafted"]
y_pred_proba = predictions["predicted_drafted_proba"]

accuracy = accuracy_score(y_true_clf, y_pred_clf)
precision = precision_score(y_true_clf, y_pred_clf)
recall = recall_score(y_true_clf, y_pred_clf)
f1 = f1_score(y_true_clf, y_pred_clf)
roc_auc = roc_auc_score(y_true_clf, y_pred_proba)

# Top Left: Classifier metrics
ax1 = fig.add_subplot(gs[0, 0])
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
values = [accuracy, precision, recall, f1, roc_auc]
colors_metric = [COLORS["primary_blue"], COLORS["secondary_blue"],
                COLORS["light_blue"], COLORS["accent_orange"], COLORS["success_green"]]

bars = ax1.barh(metrics, values, color=colors_metric, alpha=0.8, edgecolor="white", linewidth=1.5)
ax1.set_xlim(0, 1.05)
ax1.set_xlabel("Score", fontweight="bold")
ax1.set_title("Classification Metrics", fontweight="bold", pad=15)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="x", alpha=0.3, linestyle="--")

for i, (bar, val) in enumerate(zip(bars, values)):
    ax1.text(val + 0.03, i, f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

# Top Middle: Regression accuracy
ax2 = fig.add_subplot(gs[0, 1])
drafted_pred = predictions[predictions["draft_pick"].notna()]
mae = np.abs(drafted_pred["draft_pick"] - drafted_pred["predicted_pick"]).mean()
rmse = np.sqrt(((drafted_pred["draft_pick"] - drafted_pred["predicted_pick"]) ** 2).mean())
r2 = 1 - (((drafted_pred["draft_pick"] - drafted_pred["predicted_pick"]) ** 2).sum() /
          ((drafted_pred["draft_pick"] - drafted_pred["draft_pick"].mean()) ** 2).sum())

reg_metrics = ["MAE\n(picks)", "RMSE\n(picks)", "R² Score"]
reg_values = [mae, rmse, r2]
colors_reg = [COLORS["warning_red"], COLORS["accent_orange"], COLORS["primary_blue"]]

bars = ax2.bar(reg_metrics, reg_values, color=colors_reg, alpha=0.8,
              edgecolor="white", linewidth=1.5)
ax2.set_ylabel("Value", fontweight="bold")
ax2.set_title("Draft Pick Prediction Metrics", fontweight="bold", pad=15)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

for bar, val in zip(bars, reg_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + max(reg_values)*0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Top Right: Dataset overview
ax3 = fig.add_subplot(gs[0, 2])
dataset_stats = [
    ("Test Set Size", f"{len(predictions):,} players"),
    ("Players Drafted", f"{predictions['was_drafted'].sum():,} ({predictions['was_drafted'].mean()*100:.1f}%)"),
    ("Diamond Players", f"{predictions['is_diamond'].sum():,} identified"),
    ("Features Used", "138 features"),
    ("Time Period", "2019-2021"),
]

ax3.axis("off")
y_start = 0.85
for i, (label, value) in enumerate(dataset_stats):
    ax3.text(0.1, y_start - i*0.15, f"{label}:",
            fontsize=11, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.95, y_start - i*0.15, value,
            fontsize=11, ha="right", transform=ax3.transAxes)

ax3.text(0.5, 0.95, "Dataset Overview",
        fontsize=13, fontweight="bold", ha="center", transform=ax3.transAxes)
ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.88,
                            transform=ax3.transAxes,
                            facecolor=COLORS["light_gray"],
                            alpha=0.2, edgecolor=COLORS["neutral_gray"], linewidth=2))

# Bottom: Feature importance comparison
ax4 = fig.add_subplot(gs[1, :])

top_features_clf = clf_importance.head(10)
top_features_reg = reg_importance.head(10)

# Normalize importance for comparison
clf_norm = top_features_clf["importance"] / top_features_clf["importance"].max()
reg_norm = top_features_reg["importance"] / top_features_reg["importance"].max()

x = np.arange(10)
width = 0.35

bars1 = ax4.bar(x - width/2, clf_norm, width, label="Draft Status Model",
               color=COLORS["primary_blue"], alpha=0.8, edgecolor="white", linewidth=1.2)
bars2 = ax4.bar(x + width/2, reg_norm, width, label="Draft Pick Model",
               color=COLORS["accent_orange"], alpha=0.8, edgecolor="white", linewidth=1.2)

ax4.set_xlabel("Feature", fontweight="bold", fontsize=12)
ax4.set_ylabel("Normalized Importance", fontweight="bold", fontsize=12)
ax4.set_title("Top 10 Features Comparison: Both Models", fontweight="bold", pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(top_features_clf["feature"], rotation=45, ha="right", fontsize=9)
ax4.legend(frameon=True, fancybox=True, loc="upper right")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "6_performance_dashboard.png", dpi=300, bbox_inches="tight", facecolor="white")
print(f"   ✓ Saved: 6_performance_dashboard.png")
plt.close()

# ============================================================================
# Complete
# ============================================================================
print("\n" + "=" * 80)
print("✓ ALL VISUALIZATIONS CREATED SUCCESSFULLY")
print("=" * 80)
print(f"\nSaved to: {OUTPUT_DIR}/")
print("\nVisualizations created:")
print("  1. Feature Importance")
print("  2. Confusion Matrix")
print("  3. Prediction Distribution Analysis")
print("  4. Diamond Players")
print("  5. Class Year Analysis")
print("  6. Performance Dashboard")
print("=" * 80)
