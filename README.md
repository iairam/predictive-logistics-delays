# Predicting Logistics Delays: Operational Thresholds vs. Machine Learning

## Project Overview
Supply chain managers often rely on informal rules—such as flagging shipments when asset utilization crosses 90%—to predict delivery delays. Using a dataset of 1,000 logistics transactions (where 56.6% of shipments experienced delays), I tested whether standalone operational thresholds actually predict delays, and compared them against multivariate machine learning models.

---

## The Tech Stack
- **Language & Libraries:** Python (Pandas, NumPy, Scikit-Learn, Statsmodels, XGBoost, Imbalanced-Learn)
- **Statistical & ML Methods:** Logit Regression (Odds Ratios), Logistic Regression, Random Forest, XGBoost, SMOTE
- **Visuals:** Matplotlib, Seaborn

---

## Repository Structure
- `/data/` -> Smart Logistics Supply Chain dataset (1,000 transactions, 16 operational attributes)
- `/scripts/` -> Jupyter Notebook containing feature engineering, SMOTE sampling, statistical tests, and ML models
- `/visuals/` -> Charts showing delay rate thresholds, ROC curves, confusion matrix, and feature importance

---

## Data Prep & Feature Engineering
To test both simple rules and complex interactions, I engineered new features from the raw logistics logs:
- **Threshold Flags:** Created `High_Utilization` (top 10%, >95.9% capacity) and `Big_Surprise` (top 25% demand forecast error, >201.0 units), plus an interaction term multiplying both together.
- **Interaction & Traffic Features:** Applied a log transformation to waiting times (`Waiting_Time_Log`), built interaction features (`Utilization_Waiting` and `Surprise_Inventory`), and encoded traffic conditions (`Traffic_Status_Heavy` and `Traffic_Status_Detour`).
- **Preventing Data Leakage:** Split the data into 60% training, 20% validation, and 20% testing using stratified sampling. Applied SMOTE oversampling **only** to the training set so the validation and test sets reflected real-world delay rates.

---

## Key Analytical Findings

### 1. Standalone Thresholds Are Not Enough
Testing single rules with a `statsmodels` Logit model proved that simple cutoffs miss the bigger picture:
- **High Utilization (>95.9%):** Increases delay odds by **1.70×** (*p* = 0.040), pushing the delay rate from 56.0% to 62.2%.
- **Big Demand Surprises (>201.0):** Has no significant standalone effect on delays (Odds Ratio = 1.00, *p* = 0.989).
- **Combined Effect:** When high utilization and big demand surprises happened together, the delay rate unexpectedly dropped to **40.9%** (Odds Ratio = 0.32, *p* = 0.029), suggesting teams actively deploy contingency buffers during extreme stress.

### 2. Machine Learning Model Comparison (Test Set)
I trained three classification models on the 12-feature dataset and evaluated them on 200 held-out test shipments:

| Model | Test AUC | Test F1-Score | Notes |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.817** | **0.761** | Best balance of accuracy and clear business interpretability |
| **Random Forest** | 0.801 | 0.767 | Strong F1-score, slightly lower discriminative power |
| **XGBoost** | 0.825 | 0.746 | Highest AUC, lower precision/recall balance |

- **Confusion Matrix Breakdown:** The Logistic Regression model accurately classified **156 out of 200** test shipments (86 true negatives, 70 true positives) with only **1 false positive**, preventing false-alarm fatigue for operations teams.

### 3. Top Drivers of Logistics Delays
Extracting the Odds Ratios (OR) from the Logistic Regression model revealed the exact factors driving delays:
- **Heavy Traffic (`Traffic_Status_Heavy`):** Multiplies delay odds by **14.67×** compared to clear traffic. External road conditions outweigh internal warehouse metrics.
- **Demand Forecast Errors (`Demand_Surprise`):** Each unit increase raises delay odds by **1.72×**.
- **High Capacity Threshold (`High_Utilization`):** Increases delay odds by **1.61×**.
- **Queue Strain (`Utilization_Waiting`):** Combining high utilization with staging wait times raises delay odds by **1.53×**.
- **Normal Asset Utilization (`Asset_Utilization`):** Under normal running conditions, higher continuous utilization actually reduces delay odds by **~49%** (OR = 0.51), reflecting smooth operational flow.

---

## Recommendations for Supply Chain Teams
1. **Invest in Real-Time Traffic Routing:** Because heavy traffic increases delay odds by nearly 15×, teams should prioritize live GPS routing and dynamic scheduling over static warehouse rules.
2. **Replace Static Cutoffs with Risk Scoring:** Move away from simple 90% utilization alerts and deploy a multivariate risk-scoring model that flags high-risk combinations (like heavy traffic + staging wait times).
3. **Audit Contingency Buffers:** Study how teams successfully kept delays at 40.9% during simultaneous high-utilization and demand-surprise events, and apply those buffer strategies across standard operations.
