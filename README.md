# Quantitative XAU/USD Session Strategy

![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/SoloShun)

A quantitative research project to model and predict the session dynamics of Gold (XAU/USD). This repository explores multiple hypotheses using a dual modeling approach, benchmarking **Gradient Boosted Trees (XGBoost)** against **Deep Learning models (LSTM, Transformers)** for each research question.

The project investigates both **classification** (predicting market direction) and **regression** (predicting market return) to build a comprehensive, backtested understanding of intra-day market behavior.

## Project Goals & Research Questions

The primary goal of this project is to move beyond simple discretionary trading patterns and answer key quantitative questions through a rigorous, data-driven framework:

1.  **Predictive Power of Sessions:** Can the characteristics of one trading session (e.g., Asia) reliably predict the behavior of a subsequent session (e.g., London or New York)?
2.  **Intra-day Momentum:** Can the price action of the London morning session be used to forecast the high-volume London/New York overlap?
3.  **Model Benchmarking:** Which model architecture—XGBoost on tabular features or Deep Learning on sequential data—is better suited for predicting these market dynamics?

## Tech Stack

- **Data Acquisition:** MetaTrader5 API
- **Data Manipulation:** Pandas, NumPy
- **Feature Engineering:** pandas_ta
- **Modeling:** Scikit-learn, XGBoost, TensorFlow/Keras, PyTorch
- **Backtesting:** backtesting.py
- **Deployment/Demo:** Hugging Face Hub (Models & Spaces), Next.js, FastAPI

## Repository Structure

```
Quantitative-XAUUSD-Strategy/
│
├── data/ # Raw and processed datasets
├── models/ # Trained models (subfolders for xgb, lstm, etc.)
├── notebooks/ # Exploratory data analysis
├── reports/ # Backtest plots and results
├── src/ # Main source code for the workflow
└── app/ # Code for the Hugging Face Spaces demo
```

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/soloeinsteinmit/Quantitative-XAUUSD-Strategy.git
    cd Quantitative-XAUUSD-Strategy
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the data pipeline (for Hypothesis A):**
    - Update your MetaTrader 5 credentials in `src/data_acquisition.py`.
    - Execute the scripts in order to generate features, train models, and run a backtest for the "Asia -> London" hypothesis.
    ```bash
    python src/data_acquisition.py
    python src/feature_engineering.py
    python src/train_model.py # This script will handle training multiple model types
    python src/backtest.py
    ```

## Research Roadmap & Status

- [ ] **Hypothesis A (Asia -> London):** Implement and benchmark XGBoost vs. DL models.
- [ ] **Hypothesis B (London Morning -> Overlap):** Implement and benchmark all model types.
- [ ] **Hypothesis C (Asia+London -> NY):** Implement and benchmark all model types.
- [ ] **Final Analysis:** Write a comparative report on the performance of all models across all hypotheses.
- [ ] **Incorporate Additional Features:** Explore data from the DXY (US Dollar Index) and major market futures.

## Author

- **Solomon Eshun**
  - [LinkedIn](https://www.linkedin.com/in/solomon-eshun-788568177/)
  - [Medium](https://soloshun.medium.com/)
