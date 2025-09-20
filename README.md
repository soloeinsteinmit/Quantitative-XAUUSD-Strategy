# Quantitative XAU/USD Session Strategy

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/SoloShun)

A quantitative research project to model and predict the session dynamics of Gold (XAU/USD). This repository explores multiple hypotheses using both **classification** (predicting market direction), **regression** (predicting market return), and **deep learning** (LSTM & Transformer) models, with the goal of building a fully backtested algorithmic trading strategy.

## Project Goals & Research Questions

The primary goal of this project is to move beyond simple discretionary trading patterns and answer key quantitative questions through a rigorous, data-driven framework:

1.  **Predictive Power of Sessions:** Can the characteristics of one trading session (e.g., Asia) reliably predict the behavior of a subsequent session (e.g., London)?
2.  **Intra-day Momentum:** Can the price action of the London morning session be used to forecast the high-volume London/New York overlap?
3.  **Direction vs. Magnitude:** Which is more feasible to predict with machine learning: the _direction_ of a market move (up/down) or the _magnitude_ of that move (the percentage return)?

## Tech Stack

- **Data Acquisition:** MetaTrader5 API
- **Data Manipulation:** Pandas, NumPy
- **Feature Engineering:** pandas_ta
- **Modeling:** Scikit-learn, XGBoost, Deep Learning (LSTM, Transformers)
- **Backtesting:** backtesting.py
- **Deployment/Demo:** Hugging Face Hub (Models & Spaces), Next.js, FastAPI

## Repository Structure

```
Quantitative-XAUUSD-Strategy/
│
├── data/ # Raw and processed datasets
├── models/ # Trained models
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
    - Execute the scripts in order. These scripts will generate the features, train the models, and run a backtest for the "Asia -> London" prediction hypothesis.
    ```bash
    python src/data_acquisition.py
    python src/feature_engineering.py
    python src/train_model.py
    python src/backtest.py
    ```

## Backtesting Results (Hypothesis A - Initial Model)

This section will be updated as models are trained. The goal is to evaluate each hypothesis against key performance metrics.

_(Results for the initial XGBoost model will be placed here)_

| Metric        | Value |
| ------------- | :---: |
| Sharpe Ratio  |  TBD  |
| Max. Drawdown |  TBD  |
| Win Rate      |  TBD  |
| Profit Factor |  TBD  |

## Future Work & Research Roadmap

- [ ] Complete implementation and backtesting for **Hypothesis A (Asia -> London)**.
- [ ] Implement and test **Hypothesis B (London Morning -> LON/NY Overlap)**.
- [ ] Write a comparative analysis of the performance of each hypothesis.
- [ ] Experiment with LSTM and Transformer models to capture more complex time-series dependencies.
- [ ] Incorporate additional features, such as data from the DXY (US Dollar Index) and major market futures.
- [ ] Develop a more sophisticated risk management system (e.g., dynamic stop-loss based on volatility predictions).

## Author

- **Solomon Eshun**
  - [LinkedIn](https://www.linkedin.com/in/solomon-eshun-788568177/)
  - [Medium](https://soloshun.medium.com/)
