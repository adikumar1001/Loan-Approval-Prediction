# 💳 Loan Default Prediction (R · caret)

Predict whether a **loan applicant will default** using supervised machine learning. This repo captures an **end‑to‑end** workflow: data cleaning → feature engineering → model training & tuning → evaluation and business interpretation.

> Target: `loan_status` (1 = default, 0 = non‑default). Dataset size ≈ **45,000** applications with demographic, financial, and loan‑specific features.


---

## 🔎 Executive Summary

- **Best overall model:** **Random Forest** (ranger) with strong **precision** and balanced **sensitivity/specificity** on validation/test.  
- **GBM** delivered the **highest AUC** (~0.972) but showed slightly more variance across folds.  
- **KNN** served as a baseline; performance was lower and more sensitive to scaling/noise.

**Headline metrics** (approximate, from the project report):
| Model | Split | Accuracy | Kappa | Sensitivity | Specificity | AUC | Precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | Train | 92.6% | 0.77 | — | — | — | — |
| Random Forest | Val | **93.3%** | **0.79** | **78.7%** | **97.4%** | — | **89.8%** |
| Random Forest | Test | **92.9%** | **0.79** | **78.0%** | **97.0%** | — | **89.8%** |
| GBM | Val | ~80% | ~0.73 | — | — | **0.9716** | — |
| GBM | Test | ~80–83% | ~0.73 | — | — | **0.9716** | — |
| KNN | Val | ~79% | ~0.68 | — | — | ~0.86 | — |
| KNN | Test | ~78% | ~0.67 | — | — | **0.9463** | — |

> Business takeaway: flag **high‑risk** applicants early to reduce defaults while keeping approvals efficient and fair.


---

## 🗂️ Repository Contents
- `Loan default prediction.Rmd` — end‑to‑end analysis and modeling in R.
- `Project Report.pdf` — narrative write‑up with methodology, charts, and results.
- *(Optional)* Add a `data/` folder with your raw/cleaned CSV if you plan to make the repo fully reproducible.

---

## ⚙️ How to Run

1. Open the **R Markdown** file in RStudio:  
   `Loan default prediction.Rmd`
2. Install (once) and load packages:
   ```r
   install.packages(c("tidyverse","caret","ranger","gbm","pROC","kknn"))
   library(tidyverse); library(caret); library(ranger); library(gbm); library(pROC); library(kknn)
   ```
3. Knit the Rmd (▶ Knit) to HTML/PDF to reproduce preprocessing, training, and evaluation.
4. If you add a dataset, update the file path(s) at the top of the Rmd and re‑knit.

> Tip: For strict reproducibility, set a seed: `set.seed(42)` before partitioning & training.


---

## 🧹 Data Prep (highlights)

- Drop biologically implausible ages (e.g., **age > 100**).  
- Remove zero‑variance and near‑zero variance predictors (`caret::nearZeroVar`).  
- Impute/clean missing values with simple rules (e.g., income, employment length).  
- Convert categoricals to dummies; drop IDs and strictly non‑predictive fields.  
- **Split:** 70% train (≈31,499), 15% validation (≈6,750), 15% test (≈6,751).  
- Apply **min‑max normalization** for KNN only (`preProcess(method = "range")`).

---

## 🧠 Models

All models trained with **caret** and cross‑validation:

- **Random Forest** (`method = "ranger"`): tuned with `mtry = 13`, `splitrule = "gini"`; robust to mixed features and noise.  
- **GBM** (`method = "gbm"`): 5‑fold CV; grid‑tuned shrinkage, trees, and interaction depth; excels on structured data.  
- **KNN** (`method = "knn"`): normalized features; `tuneLength = 20`, repeated CV; simple, interpretable baseline.

Common evaluation: **Accuracy, Kappa, Sensitivity, Specificity, Precision, ROC‑AUC** (`pROC`).


---

## 📈 Key Visuals (from the analysis)

- **Interest rate distribution** → spike near ~10%; right‑skewed tail.  
- **Loan amount histogram** (below 99th pct) → right‑skew with peaks at \$5k, \$10k (standard tiers).  
- **Income vs Loan Status** (boxplot) → higher incomes for non‑defaulters.  
- **Interest Rate vs Status** (boxplot) → defaulters carry **higher rates**, consistent with risk‑based pricing.  
- **ROC curves** for GBM and KNN show strong separability; GBM AUC ≈ **0.972**.

---

## 🧭 Interpretation & Risk Use

- **Random Forest** recommended for production due to **stable precision** and balanced recall/specificity.  
- **GBM** may be preferred if optimizing strictly for AUC; monitor for variance across folds.  
- **Actionable features** (examples): interest rate, income, employment length, loan amount, homeownership.

> Always assess **false positive/negative costs**; tune thresholds accordingly. Use **calibration** plots if deploying.

---

## ✅ Reuse & Extension

- Add **threshold tuning** (Youden’s J / business cost curve).  
- Add **calibration** (`caret::calibration`, `pROC::coords`).  
- Try **class weights** or **SMOTE** for any imbalance.  
- Add **explainability** (permutation importance, PDPs, SHAP via `iml`/`shapviz`).  
- Package code chunks into functions for CI‑style retraining.



---

## 👤 Author

**Aditya Kumar** — Data Science & Analytics  
Suggestions and PRs welcome!
