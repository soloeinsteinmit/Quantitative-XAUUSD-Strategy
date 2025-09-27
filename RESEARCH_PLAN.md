# Research Plan: AI-Driven Analysis of XAU/USD Session Dynamics

**Author:** Solomon Eshun
**Version:** 1.0
**Date:** September 20, 2025

## 1. Project Objective

The primary objective of this project is to systematically investigate and model the relationships between different Forex trading sessions for the XAU/USD (Gold) pair. We will test a series of hypotheses using a rigorous, data-driven approach, employing and benchmarking both **gradient boosted models** and **deep learning models**.

The ultimate goal is to develop and backtest a trading strategy based on the most promising models, creating a tangible portfolio piece that showcases skills in quantitative analysis, AI engineering, and financial market modeling.

## 2. Core Methodological Framework

While each hypothesis will have unique features and targets, the underlying methodology for development and evaluation will remain consistent.

- **Data Source:** 1-Hour (H1) historical price data for XAU/USD from 2018-Present, acquired via the MetaTrader 5 Python API.
- **Modeling Approach:** We will benchmark two families of models for each hypothesis:
  1.  **Gradient Boosted Models:** XGBoost (`XGBClassifier` for direction and `XGBRegressor` for return) as a powerful baseline for structured, tabular data.
  2.  **Deep Learning Models:** Recurrent Neural Networks (LSTMs) and Transformers to capture complex temporal patterns and sequential dependencies.
- **Backtesting Framework:** The `backtesting.py` library.
  - **Transaction Costs:** 0.02% commission to simulate spread and fees.
  - **Validation:** Time-based 80% train / 20% test split.
- **Evaluation Metrics:**
  - **For Classification:** Accuracy, F1-Score, Sharpe Ratio, Profit Factor.
  - **For Regression:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Sharpe Ratio of a strategy trading on the predicted return.

## 3. Key Session Definitions (UTC)

- **Asian Session:** 22:00 (previous day) - 07:59
- **London Session:** 08:00 - 16:59
- **New York Session:** 17:00 - 21:59
- **London Morning:** 08:00 - 12:59
- **LON/NY Overlap:** 13:00 - 16:59

---

## 4. Research Hypotheses

### Hypothesis A: Asia -> London Prediction

- **Hypothesis Statement:** The characteristics of the Asian session contain predictive information about the subsequent London session's behavior.

#### A1: Predicting London Direction (Classification)

- **Target (`y`):** `london_direction` -> `1` if London Close > London Open, else `0`.

#### A2: Predicting London Return (Regression)

- **Target (`y`):** `london_return` -> The percentage return of the London session.

---

### Hypothesis B: London Morning -> Overlap Prediction (The "Mid-Day Momentum" Model)

- **Hypothesis Statement:** The price action of the London morning session provides strong predictive signals for the highly liquid LON/NY Overlap session.

#### B1: Predicting Overlap Direction (Classification)

- **Target (`y`):** `overlap_direction` -> `1` if Overlap Close > Overlap Open, else `0`.

#### B2: Predicting Overlap Return (Regression)

- **Target (`y`):** `overlap_return` -> The percentage return of the Overlap session.

#### B3: Predicting Overlap Volatility Spike (Classification)

- **Target (`y`):** `is_overlap_spike` -> `1` if the overlap range exceeds a dynamic threshold.

---

### Hypothesis C: (Asia + London) -> New York Prediction (The "Closing Bell" Model)

- **Hypothesis Statement:** The combined characteristics of both the Asian and full London sessions contain predictive information about the subsequent New York session.

#### C1: Predicting New York Direction (Classification)

- **Problem Type:** Binary Classification.
- **Feature Set (`X` - Known at 16:59 UTC):** This includes all features from Hypothesis A, plus new features calculated from the full London session (`london_full_return`, `london_full_range`, technical indicators updated to 16:59 UTC).
- **Target Variable (`y`):** `ny_direction` -> `1` if New York Close (21:00) > New York Open (17:00), else `0`.

#### C2: Predicting New York Return (Regression)

- **Problem Type:** Regression.
- **Feature Set (`X`):** Identical to C1.
- **Target Variable (`y`):** `ny_return` -> The percentage return `(Close_21:00 - Open_17:00) / Open_17:00`.

<!-- ---

## 5. Project Workflow

1.  **Implement Hypothesis A:** Build the full data pipeline and train/backtest both XGBoost and Deep Learning models.
2.  **Document and Share (v1):** Share the results, code, and models for Hypothesis A on GitHub and Hugging Face, accompanied by a Medium/LinkedIn article comparing the performance of the different model architectures.
3.  **Implement Hypothesis B & C:** Adapt the existing codebase to test the remaining hypotheses, benchmarking both model types for each.
4.  **Document and Share (v2):** Write a follow-up article comparing the predictive power across all three hypotheses, providing a final conclusion on which session dynamics are most predictable. -->
