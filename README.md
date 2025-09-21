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

2.  **Set up Python environment:**

    ```bash
    # Option 1: Using venv
    python -m venv trade_env
    # On Windows
    .\trade_env\Scripts\activate
    # On Unix or MacOS
    source trade_env/bin/activate

    # Install requirements
    pip install -r requirements.txt
    ```

    ```bash
    # Option 2: Using conda
    conda create -n trade_env python=3.12
    conda activate trade_env

    # Install requirements
    pip install -r requirements.txt
    ```

3.  **Configure MetaTrader 5:**

    - Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

    - Update the `.env` file with your MetaTrader 5 credentials:

    ```env
    DEMO_ACCOUNT_NUMBER=YOUR_ACCOUNT_NUMBER
    PASSWORD=YOUR_PASSWORD
    SERVER=MetaQuotes-Demo
    ```

4.  **Run the pipeline:**

    Option 1: Step by Step (Recommended for first run)

    ```bash
    # 1. Download data from MetaTrader 5
    python src/data_acquisition.py --year 2023

    # 2. Generate features for Hypothesis A
    python src/hyp_a_feature_engineering.py --year 2023 --timeframe h1 --symbol xauusd

    # 3. Train and evaluate models
    python src/hyp_a_train_model.py --year 2023 --hypothesis hyp_a
    ```

    Option 2: Using the Interactive Pipeline

    ```bash
    # Using Python
    python src/run_pipeline.py
    ```

    The interactive pipeline will prompt for parameters and run all steps automatically.

5.  **View Results:**
    - Model performance metrics and analysis: `reports/model_results_*.txt`
    - Pipeline execution logs: `reports/report_*.log`
    - Trained models: `models/xgb_classifier_*.joblib` and `models/xgb_regressor_*.joblib`

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
