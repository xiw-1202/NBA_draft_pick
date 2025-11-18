# 🏀 NBA Draft Prediction Project

This project predicts NBA draft outcomes using advanced college basketball statistics (2009–2021). The goal is not only to forecast draft selections and draft position but also to identify undervalued players who later overperform in the NBA — also known as “diamond” players.

---

## 🎯 Project Objectives

- **Classification**: Predict whether a player will be drafted
- **Regression**: Predict the exact draft pick number (1–60)
- **Diamond Detection**: Identify undervalued prospects who become strong NBA contributors

---

## 🧠 Key Concepts

- Incorporate **efficiency**, **usage**, **shooting**, and **rebounding** metrics from college
- Evaluate organizational context using **team rankings** and **conference strength**
- Benchmark career success using **NBA RAPTOR analytics**
- Use machine learning pipelines for robust evaluation

---

## 📁 Repository Structure

```
DataLab/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned + merged data
│   └── final/            # Model-ready datasets
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── models.py
├── models/               # Saved model artifacts
├── results/              # Visualizations + reports
├── requirements.txt      # Python dependencies
├── CLAUDE.md            # Detailed implementation guide
└── README.md
```

---

## 📊 Data Sources

| Source                               | Description                                         |
| ------------------------------------ | --------------------------------------------------- |
| College Basketball Stats (2009–2021) | Advanced performance stats for each player’s season |
| NBA Draft History                    | Round, overall pick, and drafting team              |
| Team Rankings                        | Contextual competitiveness measures                 |
| NBA RAPTOR Metrics                   | Career value signal for diamond detection           |

---

## 🚀 Modeling Pipeline Overview

### Stage 1 — Data Integration & Cleaning

- Standardize player names and ensure correct merges
- Final college season identification
- Merge draft results + team context

### Stage 2 — Feature Engineering

- Per-game and per-36-minute stats
- Efficiency metrics (TS%, eFG%, turnover rates)
- Conference & team strength as contextual features
- Experience and career stage signals (year, age proxy)

### Stage 3 — Data Validation & Preprocessing

- Time-based train/test split for realistic evaluation
- Scaling + encoding with sklearn Pipelines
- Check distributions, missingness, and outliers

### Stage 4 — Model Training & Evaluation

- Draft likelihood model (classification)
- Draft pick predictor (regression)
- Bayesian hyperparameter tuning
- Feature importance analysis

### Stage 5 — Diamond Player Discovery

- Connect predictions with career RAPTOR scores
- Identify undervalued players post-draft
- Generate insights to improve scouting models

---

## 📈 Performance Targets

| Objective         | Target Metric      | Goal                 |
| ----------------- | ------------------ | -------------------- |
| Draft prediction  | F1-Score           | > 0.70               |
| Draft pick error  | MAE                | < 10 picks           |
| Diamond detection | # meaningful finds | 5–10 per draft class |

---

## 🔮 Future Development

- Feature segmentation by player position (G/F/C modeling)
- NCAA tournament & combine data integration
- International + transfer portal adjustments
- Ensemble models to improve draft slot regression
- Scouting-style report generation for predictions

---

## 📦 Requirements

### Core Dependencies

- **Python**: 3.9+ (3.10 recommended)
- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn, lightgbm, optuna
- **Visualization**: matplotlib, seaborn
- **Utilities**: fuzzywuzzy, python-Levenshtein, joblib, openpyxl

See `requirements.txt` for specific versions.

---

## 🧪 Getting Started

```bash
# Clone the repository
git clone https://github.com/xiw-1202/NBA_draft_pick.git
cd NBA_draft_pick

# Create a fresh conda environment (recommended)
conda create -n nba python=3.10 -y
conda activate nba

# Install essential packages
pip install -r requirements.txt

# Run data processing and modeling steps
python src/data_processing.py
python src/feature_engineering.py
python src/models.py
```

---

## 📚 Documentation

For detailed implementation instructions, refer to [CLAUDE.md](CLAUDE.md) which contains:
- Granular step-by-step implementation guide
- Data quality checks and validation procedures
- Model development and evaluation protocols
- Diamond player discovery methodology

---

## 🤝 Contributing

This is an academic research project. For questions or collaboration inquiries, please open an issue on GitHub.

---

## 📄 License

This project is for educational purposes. Data sources remain property of their respective owners.
