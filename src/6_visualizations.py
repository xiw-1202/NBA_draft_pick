"""
Complete Visualization Script - NBA Draft Prediction Results
=============================================================
Creates comprehensive visualizations of model performance and insights.

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

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# ============================================================================
# Configuration
# ============================================================================
# Use relative paths for portability
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUT_DIR = PROJECT_ROOT / "Visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CREATING NBA DRAFT PREDICTION VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\n1️⃣ Loading data...")

predictions = pd.read_csv(MODELS_DIR / "test_predictions.csv")
clf_importance = pd.read_csv(MODELS_DIR / "classifier_feature_importance.csv")
reg_importance = pd.read_csv(MODELS_DIR / "regressor_feature_importance.csv")

print(f"   ✅ Loaded predictions: {len(predictions):,} players")
print(f"   ✅ Loaded feature importance data")

# ============================================================================
# 1. Feature Importance Comparison
# ============================================================================
print("\n2️⃣ Creating feature importance plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Classifier importance
top_clf = clf_importance.head(15)
axes[0].barh(range(len(top_clf)), top_clf["importance"], color="steelblue")
axes[0].set_yticks(range(len(top_clf)))
axes[0].set_yticklabels(top_clf["feature"])
axes[0].invert_yaxis()
axes[0].set_xlabel("Importance Score", fontsize=11)
axes[0].set_title(
    "Top 15 Features for Draft Status Prediction", fontsize=12, fontweight="bold"
)
axes[0].grid(axis="x", alpha=0.3)

# Regressor importance
top_reg = reg_importance.head(15)
axes[1].barh(range(len(top_reg)), top_reg["importance"], color="coral")
axes[1].set_yticks(range(len(top_reg)))
axes[1].set_yticklabels(top_reg["feature"])
axes[1].invert_yaxis()
axes[1].set_xlabel("Importance Score", fontsize=11)
axes[1].set_title(
    "Top 15 Features for Draft Pick Prediction", fontsize=12, fontweight="bold"
)
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_feature_importance.png", dpi=300, bbox_inches="tight")
print(f"   ✅ Saved: 1_feature_importance.png")
plt.close()

# ============================================================================
# 2. Confusion Matrix Heatmap
# ============================================================================
print("\n3️⃣ Creating confusion matrix...")

from sklearn.metrics import confusion_matrix

y_true = predictions["was_drafted"]
y_pred = predictions["predicted_drafted"]
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Undrafted", "Drafted"],
    yticklabels=["Undrafted", "Drafted"],
    annot_kws={"size": 14},
)
ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
ax.set_title(
    "Draft Status Prediction - Confusion Matrix", fontsize=14, fontweight="bold"
)

# Add annotations
total = cm.sum()
accuracy = (cm[0, 0] + cm[1, 1]) / total
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
plt.text(
    0.5,
    -0.15,
    f"Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}",
    ha="center",
    transform=ax.transAxes,
    fontsize=11,
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_confusion_matrix.png", dpi=300, bbox_inches="tight")
print(f"   ✅ Saved: 2_confusion_matrix.png")
plt.close()

# ============================================================================
# 3. IMPROVED Prediction Distribution (Multiple Views)
# ============================================================================
print("\n4️⃣ Creating improved prediction distribution...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

drafted_probs = predictions[predictions["was_drafted"] == 1]["predicted_drafted_proba"]
undrafted_probs = predictions[predictions["was_drafted"] == 0][
    "predicted_drafted_proba"
]

# 3a. Drafted players only (top left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(drafted_probs, bins=30, color="green", edgecolor="black", alpha=0.7)
ax1.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Threshold (0.5)")
ax1.set_xlabel("Predicted Draft Probability", fontsize=11, fontweight="bold")
ax1.set_ylabel("Count", fontsize=11, fontweight="bold")
ax1.set_title(
    f"Actually Drafted Players (n={len(drafted_probs)})", fontsize=12, fontweight="bold"
)
ax1.legend()
ax1.grid(alpha=0.3)
ax1.text(
    0.05,
    0.95,
    f"Mean: {drafted_probs.mean():.3f}\nMedian: {drafted_probs.median():.3f}",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)

# 3b. Undrafted players only (top right)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(undrafted_probs, bins=30, color="red", edgecolor="black", alpha=0.7)
ax2.axvline(0.5, color="darkred", linestyle="--", linewidth=2, label="Threshold (0.5)")
ax2.set_xlabel("Predicted Draft Probability", fontsize=11, fontweight="bold")
ax2.set_ylabel("Count", fontsize=11, fontweight="bold")
ax2.set_title(
    f"Not Drafted Players (n={len(undrafted_probs)})", fontsize=12, fontweight="bold"
)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.text(
    0.05,
    0.95,
    f"Mean: {undrafted_probs.mean():.3f}\nMedian: {undrafted_probs.median():.3f}",
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
)

# 3c. Overlapping with log scale (middle left)
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(
    undrafted_probs,
    bins=30,
    alpha=0.6,
    label="Not Drafted",
    color="red",
    edgecolor="black",
)
ax3.hist(
    drafted_probs,
    bins=30,
    alpha=0.8,
    label="Actually Drafted",
    color="green",
    edgecolor="black",
)
ax3.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Threshold")
ax3.set_xlabel("Predicted Draft Probability", fontsize=11, fontweight="bold")
ax3.set_ylabel("Count (Log Scale)", fontsize=11, fontweight="bold")
ax3.set_yscale("log")
ax3.set_title("Distribution Comparison (Log Scale)", fontsize=12, fontweight="bold")
ax3.legend()
ax3.grid(alpha=0.3)

# 3d. Box plot comparison (middle right)
ax4 = fig.add_subplot(gs[1, 1])
box_data = [undrafted_probs, drafted_probs]
bp = ax4.boxplot(
    box_data, labels=["Not Drafted", "Drafted"], patch_artist=True, widths=0.6
)
bp["boxes"][0].set_facecolor("red")
bp["boxes"][1].set_facecolor("green")
for box in bp["boxes"]:
    box.set_alpha(0.7)
ax4.axhline(0.5, color="black", linestyle="--", linewidth=2, alpha=0.5)
ax4.set_ylabel("Predicted Draft Probability", fontsize=11, fontweight="bold")
ax4.set_title("Distribution Summary (Box Plot)", fontsize=12, fontweight="bold")
ax4.grid(alpha=0.3, axis="y")

# 3e. Draft pick accuracy scatter (bottom left)
ax5 = fig.add_subplot(gs[2, 0])
drafted = predictions[predictions["draft_pick"].notna()]
scatter = ax5.scatter(
    drafted["draft_pick"],
    drafted["predicted_pick"],
    alpha=0.6,
    s=50,
    c=drafted["draft_pick"],
    cmap="viridis",
    edgecolors="black",
    linewidths=0.5,
)
ax5.plot([1, 60], [1, 60], "r--", linewidth=2, label="Perfect Prediction")
ax5.set_xlabel("Actual Draft Pick", fontsize=11, fontweight="bold")
ax5.set_ylabel("Predicted Draft Pick", fontsize=11, fontweight="bold")
ax5.set_title("Draft Pick Prediction Accuracy", fontsize=12, fontweight="bold")
ax5.legend()
ax5.grid(alpha=0.3)

mae = np.abs(drafted["draft_pick"] - drafted["predicted_pick"]).mean()
rmse = np.sqrt(((drafted["draft_pick"] - drafted["predicted_pick"]) ** 2).mean())
ax5.text(
    0.05,
    0.95,
    f"MAE: {mae:.2f} picks\nRMSE: {rmse:.2f}",
    transform=ax5.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
)

plt.colorbar(scatter, ax=ax5, label="Actual Pick")

# 3f. Calibration plot (bottom right)
ax6 = fig.add_subplot(gs[2, 1])

predictions["confidence_bin"] = pd.cut(
    predictions["predicted_drafted_proba"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
)

conf_accuracy = (
    predictions.groupby("confidence_bin", observed=True)
    .agg({"was_drafted": ["sum", "count", "mean"]})
    .reset_index()
)
conf_accuracy.columns = ["Confidence", "Drafted", "Total", "Actual_Rate"]

x_pos = range(len(conf_accuracy))
bars = ax6.bar(
    x_pos,
    conf_accuracy["Actual_Rate"] * 100,
    color=["darkred", "orange", "yellow", "lightgreen", "darkgreen"],
    edgecolor="black",
    alpha=0.7,
)

for i, (idx, row) in enumerate(conf_accuracy.iterrows()):
    ax6.text(
        i,
        row["Actual_Rate"] * 100 + 2,
        f"n={row['Total']:.0f}\n{row['Drafted']:.0f} drafted",
        ha="center",
        fontsize=8,
    )

ax6.set_xticks(x_pos)
ax6.set_xticklabels(conf_accuracy["Confidence"], rotation=45)
ax6.set_xlabel("Model Confidence Bin", fontsize=11, fontweight="bold")
ax6.set_ylabel("Actual Draft Rate (%)", fontsize=11, fontweight="bold")
ax6.set_title("Calibration: Predicted vs Actual", fontsize=12, fontweight="bold")
ax6.grid(alpha=0.3, axis="y")

bin_centers = [10, 30, 50, 70, 90]
ax6.plot(x_pos, bin_centers, "r--", linewidth=2, alpha=0.5, label="Perfect Calibration")
ax6.legend()

fig.suptitle(
    "Draft Prediction Analysis - Detailed Breakdown",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

plt.savefig(OUTPUT_DIR / "3_prediction_distribution.png", dpi=300, bbox_inches="tight")
print(f"   ✅ Saved: 3_prediction_distribution.png")
plt.close()

# ============================================================================
# 4. Diamond Players Visualization
# ============================================================================
print("\n5️⃣ Creating diamond players plot...")

diamonds = predictions[predictions["is_diamond"] == True]

fig, ax = plt.subplots(figsize=(12, 8))

# Plot all players (background)
ax.scatter(
    predictions["bpm"],
    predictions["raptor_total_mean"],
    alpha=0.2,
    s=30,
    c="gray",
    label="All Players",
)

# Highlight drafted players
drafted_all = predictions[predictions["was_drafted"] == 1]
ax.scatter(
    drafted_all["bpm"],
    drafted_all["raptor_total_mean"],
    alpha=0.5,
    s=50,
    c="blue",
    label="Drafted",
)

# Highlight diamonds
ax.scatter(
    diamonds["bpm"],
    diamonds["raptor_total_mean"],
    s=200,
    c="gold",
    edgecolors="red",
    linewidths=2,
    marker="*",
    label="Diamond Players",
    zorder=10,
)

# Annotate top 5 diamonds
top_diamonds = diamonds.nlargest(5, "raptor_total_mean")
for idx, row in top_diamonds.iterrows():
    ax.annotate(
        row["player_name_clean"][:15],
        (row["bpm"], row["raptor_total_mean"]),
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

ax.axhline(0, color="black", linestyle="--", alpha=0.3)
ax.axvline(0, color="black", linestyle="--", alpha=0.3)
ax.set_xlabel("College BPM (Box Plus/Minus)", fontsize=11, fontweight="bold")
ax.set_ylabel("NBA RAPTOR (Performance)", fontsize=11, fontweight="bold")
ax.set_title(
    "Diamond Players: High NBA Performance Despite Low Draft Status",
    fontsize=13,
    fontweight="bold",
)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# Add stats box
stats_text = f"Diamond Players: {len(diamonds)}\nDetection Rate: 50.0%"
ax.text(
    0.98,
    0.02,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_diamond_players.png", dpi=300, bbox_inches="tight")
print(f"   ✅ Saved: 4_diamond_players.png")
plt.close()

# ============================================================================
# 5. Class Year Analysis (Using Full Dataset)
# ============================================================================
print("\n6️⃣ Creating class year analysis...")

# Load full featured data (has 'yr' column that test_predictions lacks)
DATA_DIR = Path("/Users/sam/Documents/School/Emory/Datalab_NBA_Pick/Data/PROCESSED")
try:
    full_data = pd.read_csv(DATA_DIR / "featured_data.csv")
    # Filter to test years only (2019-2021)
    test_data_full = full_data[full_data["year"].between(2019, 2021)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "yr" in test_data_full.columns:
        # Left plot: Draft rate by class
        class_stats = (
            test_data_full.groupby("yr")
            .agg({"was_drafted": ["sum", "count", "mean"]})
            .reset_index()
        )
        class_stats.columns = ["Class", "Drafted", "Total", "Rate"]
        class_stats["Rate"] = class_stats["Rate"] * 100

        # Sort by class year
        class_order = ["Fr", "So", "Jr", "Sr"]
        class_stats["Class"] = pd.Categorical(
            class_stats["Class"], categories=class_order, ordered=True
        )
        class_stats = class_stats.sort_values("Class").reset_index(drop=True)

        colors = ["#ff9999", "#ffcc99", "#99ccff", "#99ff99"]
        axes[0].bar(
            range(len(class_stats)),
            class_stats["Rate"],
            color=colors[: len(class_stats)],
        )
        axes[0].set_xticks(range(len(class_stats)))
        axes[0].set_xticklabels(class_stats["Class"])
        axes[0].set_xlabel("Class Year", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Draft Rate (%)", fontsize=11, fontweight="bold")
        axes[0].set_title(
            "Draft Rate by Class Year (2019-2021)", fontsize=12, fontweight="bold"
        )
        axes[0].grid(axis="y", alpha=0.3)

        # Add labels
        for i, row in class_stats.iterrows():
            axes[0].text(
                i,
                row["Rate"] + 0.15,
                f"n={row['Total']:.0f}\n{row['Drafted']:.0f} drafted",
                ha="center",
                fontsize=9,
            )

        # Right plot: Average pick by class
        drafted_by_class = (
            test_data_full[test_data_full["draft_pick"].notna()]
            .groupby("yr")["draft_pick"]
            .agg(["mean", "count"])
            .reset_index()
        )
        drafted_by_class.columns = ["Class", "Avg_Pick", "Count"]
        drafted_by_class["Class"] = pd.Categorical(
            drafted_by_class["Class"], categories=class_order, ordered=True
        )
        drafted_by_class = drafted_by_class.sort_values("Class").reset_index(drop=True)

        axes[1].bar(
            range(len(drafted_by_class)),
            drafted_by_class["Avg_Pick"],
            color=colors[: len(drafted_by_class)],
        )
        axes[1].set_xticks(range(len(drafted_by_class)))
        axes[1].set_xticklabels(drafted_by_class["Class"])
        axes[1].set_xlabel("Class Year", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Average Draft Pick", fontsize=11, fontweight="bold")
        axes[1].set_title(
            "Average Draft Position by Class Year", fontsize=12, fontweight="bold"
        )
        axes[1].invert_yaxis()  # Lower = better
        axes[1].grid(axis="y", alpha=0.3)

        # Add value labels
        for i, row in drafted_by_class.iterrows():
            axes[1].text(
                i,
                row["Avg_Pick"],
                f"{row['Avg_Pick']:.1f}\n(n={row['Count']:.0f})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    else:
        # Fallback if no 'yr' column
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "Class year data not available",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_title("Class Year Analysis", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_class_year_analysis.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: 5_class_year_analysis.png")
    plt.close()

except Exception as e:
    print(f"   ⚠️  Could not create class year analysis: {e}")
    # Create placeholder
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax in axes:
        ax.text(
            0.5,
            0.5,
            "Class year data not available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_title("Class Year Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_class_year_analysis.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved placeholder: 5_class_year_analysis.png")
    plt.close()

# ============================================================================
# 6. Model Performance Dashboard
# ============================================================================
print("\n7️⃣ Creating performance dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle(
    "NBA Draft Prediction Model - Performance Dashboard",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# 6a. Classifier Metrics
ax1 = fig.add_subplot(gs[0, 0])
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
values = [0.9704, 0.3606, 0.8258, 0.5020, 0.9674]
colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
ax1.barh(metrics, values, color=colors)
ax1.set_xlim(0, 1)
ax1.set_xlabel("Score")
ax1.set_title("Classifier Performance", fontweight="bold")
ax1.grid(axis="x", alpha=0.3)
for i, v in enumerate(values):
    ax1.text(v + 0.02, i, f"{v:.3f}", va="center")

# 6b. Regressor Metrics
ax2 = fig.add_subplot(gs[0, 1])
drafted_pred = predictions[predictions["draft_pick"].notna()]
within_5 = (
    np.abs(drafted_pred["draft_pick"] - drafted_pred["predicted_pick"]) <= 5
).mean() * 100
within_10 = (
    np.abs(drafted_pred["draft_pick"] - drafted_pred["predicted_pick"]) <= 10
).mean() * 100
within_15 = (
    np.abs(drafted_pred["draft_pick"] - drafted_pred["predicted_pick"]) <= 15
).mean() * 100

accuracy_metrics = ["Within 5", "Within 10", "Within 15"]
accuracy_values = [within_5, within_10, within_15]
ax2.bar(accuracy_metrics, accuracy_values, color=["#e74c3c", "#f39c12", "#2ecc71"])
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Draft Pick Prediction Accuracy", fontweight="bold")
ax2.grid(axis="y", alpha=0.3)
for i, v in enumerate(accuracy_values):
    ax2.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

# 6c. Dataset Overview
ax3 = fig.add_subplot(gs[0, 2])
dataset_info = f"""
Dataset Statistics:
━━━━━━━━━━━━━━━━━━━
Test Set: {len(predictions):,}
Drafted: {predictions['was_drafted'].sum():,}
Draft Rate: {predictions['was_drafted'].mean()*100:.2f}%

Diamonds: {predictions['is_diamond'].sum():,}
Detection: 50.0%

Features: 139
MAE: 11.93 picks
RMSE: 15.09 picks
"""
ax3.text(
    0.1,
    0.5,
    dataset_info,
    fontsize=10,
    family="monospace",
    verticalalignment="center",
    transform=ax3.transAxes,
)
ax3.axis("off")

# 6d. Top Features (Classifier)
ax4 = fig.add_subplot(gs[1, :])
top_10_clf = clf_importance.head(10)
ax4.barh(range(10), top_10_clf["importance"], color="steelblue", alpha=0.7)
ax4.set_yticks(range(10))
ax4.set_yticklabels(top_10_clf["feature"])
ax4.invert_yaxis()
ax4.set_xlabel("Importance Score")
ax4.set_title("Top 10 Features for Draft Status Prediction", fontweight="bold")
ax4.grid(axis="x", alpha=0.3)

# 6e. Top Features (Regressor)
ax5 = fig.add_subplot(gs[2, :])
top_10_reg = reg_importance.head(10)
ax5.barh(range(10), top_10_reg["importance"], color="coral", alpha=0.7)
ax5.set_yticks(range(10))
ax5.set_yticklabels(top_10_reg["feature"])
ax5.invert_yaxis()
ax5.set_xlabel("Importance Score")
ax5.set_title("Top 10 Features for Draft Pick Prediction", fontweight="bold")
ax5.grid(axis="x", alpha=0.3)

plt.savefig(OUTPUT_DIR / "6_performance_dashboard.png", dpi=300, bbox_inches="tight")
print(f"   ✅ Saved: 6_performance_dashboard.png")
plt.close()

# ============================================================================
# 7. Error Analysis
# ============================================================================
print("\n8️⃣ Creating error analysis plot...")

drafted_with_pred = predictions[predictions["predicted_pick"].notna()].copy()
drafted_with_pred["pick_error"] = np.abs(
    drafted_with_pred["draft_pick"] - drafted_with_pred["predicted_pick"]
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error distribution
axes[0, 0].hist(
    drafted_with_pred["pick_error"], bins=30, color="skyblue", edgecolor="black"
)
axes[0, 0].axvline(
    drafted_with_pred["pick_error"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f'Mean: {drafted_with_pred["pick_error"].mean():.1f}',
)
axes[0, 0].set_xlabel("Prediction Error (picks)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Distribution of Prediction Errors", fontweight="bold")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Error by draft round
drafted_with_pred["round"] = (drafted_with_pred["draft_pick"] > 30).map(
    {True: "2nd Round", False: "1st Round"}
)
round_errors = drafted_with_pred.groupby("round")["pick_error"].mean()
axes[0, 1].bar(round_errors.index, round_errors.values, color=["gold", "silver"])
axes[0, 1].set_ylabel("Average Error (picks)")
axes[0, 1].set_title("Prediction Error by Draft Round", fontweight="bold")
axes[0, 1].grid(axis="y", alpha=0.3)

# Error vs BPM
axes[1, 0].scatter(
    drafted_with_pred["bpm"], drafted_with_pred["pick_error"], alpha=0.5, s=50
)
axes[1, 0].set_xlabel("College BPM")
axes[1, 0].set_ylabel("Prediction Error (picks)")
axes[1, 0].set_title("Prediction Error vs College Performance", fontweight="bold")
axes[1, 0].grid(alpha=0.3)

# Biggest errors
biggest_errors = drafted_with_pred.nlargest(10, "pick_error")
axes[1, 1].barh(range(10), biggest_errors["pick_error"], color="orangered")
axes[1, 1].set_yticks(range(10))
axes[1, 1].set_yticklabels([name[:20] for name in biggest_errors["player_name_clean"]])
axes[1, 1].invert_yaxis()
axes[1, 1].set_xlabel("Error (picks)")
axes[1, 1].set_title("Top 10 Prediction Errors", fontweight="bold")
axes[1, 1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "7_error_analysis.png", dpi=300, bbox_inches="tight")
print(f"   ✅ Saved: 7_error_analysis.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ VISUALIZATION COMPLETE!")
print("=" * 80)

print(f"\n📁 Saved 7 visualizations to: {OUTPUT_DIR}")
print(f"\nFiles created:")
print(f"   1. 1_feature_importance.png - Feature importance comparison")
print(f"   2. 2_confusion_matrix.png - Classification performance")
print(f"   3. 3_prediction_distribution.png - Multi-view prediction analysis")
print(f"   4. 4_diamond_players.png - Diamond player identification")
print(f"   5. 5_class_year_analysis.png - Class year insights")
print(f"   6. 6_performance_dashboard.png - Complete performance summary")
print(f"   7. 7_error_analysis.png - Prediction error breakdown")

print(f"\n🎨 All visualizations saved as high-resolution PNG files (300 DPI)")
print(f"🏀 Ready for presentations, reports, and portfolios!")
