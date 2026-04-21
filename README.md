# Trader Performance vs Market Sentiment Analysis

## Primetrade.ai — Data Science Intern (Round-0 Assignment)

### Project Overview

This project analyzes the relationship between **Bitcoin Market Sentiment (Fear/Greed Index)** and **trader behavior and performance on Hyperliquid**. The goal is to uncover patterns that could inform smarter trading strategies.

### Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| `data/fear_greed_index.csv` | Bitcoin Fear & Greed Index (daily) | ~2,646 rows |
| `data/historical_data.csv` | Hyperliquid trader-level data | ~211,225 rows |

### Project Structure

```
ASSIGN/
├── data/
│   ├── fear_greed_index.csv              # Sentiment dataset
│   └── historical_data.csv               # Trader dataset
├── charts/                                # Generated charts (auto-created)
├── Trader_Performance_vs_Market_Sentiment.ipynb     # Main Jupyter Notebook
├── Trader_Performance_vs_Market_Sentiment_executed.ipynb  # Executed notebook with outputs
├── app.py                                 # Streamlit Dashboard (Bonus)
├── generate_notebook.py                   # Notebook generation script
└── README.md                              # This file
```

### How to Run

#### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter streamlit plotly
```

#### 1. Jupyter Notebook
```bash
jupyter notebook Trader_Performance_vs_Market_Sentiment.ipynb
```
Run all cells to see the complete analysis with charts.

#### 2. Streamlit Dashboard (Bonus)
```bash
streamlit run app.py
```
Opens an interactive dashboard at `http://localhost:8501`.

#### 3. Regenerate Notebook
```bash
python generate_notebook.py
```

---

### Methodology

1. **Data Loading & Cleaning**: Loaded both datasets, converted timestamps, handled missing values, and removed duplicates.
2. **Date Alignment**: Aligned trader data with sentiment data at daily level using inner join on date.
3. **Feature Engineering**: Created daily PnL per trader, win rate, trade frequency, position sizes, long/short ratios, and PnL volatility metrics.
4. **Segmentation**: Classified traders by leverage usage, trading frequency, and PnL consistency.
5. **Analysis**: Compared all metrics across Fear vs Greed sentiment regimes.
6. **Predictive Modeling**: Built Logistic Regression and Random Forest models to predict profitable days.
7. **Clustering**: Applied K-Means to identify trader behavioral archetypes.

---

### Key Insights

#### Insight 1: Sentiment-Performance Relationship
- Trader profitability metrics (PnL, win rate) show measurable differences between Fear and Greed days.
- PnL volatility tends to be higher during Fear periods, indicating riskier trading conditions.

#### Insight 2: Behavioral Shifts
- Traders adjust their behavior based on sentiment — trade frequency, position sizing, and directional bias all shift.
- During Greed periods, there tends to be higher long-bias; Fear periods show relatively more short activity.

#### Insight 3: Segment-Specific Patterns
- High-leverage traders face amplified risks during Fear days.
- Consistent winners maintain steadier performance across both sentiment regimes.
- Frequent traders are more sensitive to sentiment shifts than infrequent traders.

---

### Strategy Recommendations

| Strategy | Fear Days | Greed Days |
|----------|-----------|------------|
| **Position Sizing** | Reduce by 20-30% | Standard / increase slightly |
| **Trade Frequency** | Reduce (quality > quantity) | Increase (more favorable conditions) |
| **Direction Bias** | Increase short exposure | Maintain long bias |
| **Segment-Specific** | Low-leverage traders should reduce activity | Frequent traders should capitalize on momentum |

---

### Tools & Libraries

- **Python 3.12**
- **pandas, numpy** – data manipulation
- **matplotlib, seaborn** – static visualizations
- **plotly** – interactive charts
- **scikit-learn** – ML models (Logistic Regression, Random Forest, K-Means)
- **Streamlit** – interactive dashboard
- **Jupyter** – notebook environment

---

### Author

Data Science Intern Assignment Submission — Primetrade.ai
