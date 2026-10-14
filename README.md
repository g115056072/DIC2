# 50 Startups — ML Feature Selection Dashboard

A multiple linear regression project on the 50 Startups dataset with comprehensive feature
selection analysis and an interactive Flask web dashboard.

Live website: https://dic-2-alpha.vercel.app/

## Features

- **RMSE Comparison** — best-subset search for 1 to 5 features
- **5 Feature Selection Methods**: LassoCV, RFECV, SFS, ElasticNetCV, Exhaustive Search
- **Interactive Web Dashboard** — dark glassmorphism UI with Chart.js visualisations
- **Live Profit Predictor** — enter startup details and get an instant profit estimate
- **Feature Selection Explorer** — switch between methods and feature counts with live RMSE/R² summaries
- **Statistical Diagnostics** — fitted equation, coefficient standard errors, confidence intervals, t-statistics, and VIF
- **Test-set Outlier Review** — ranked residuals with absolute and percentage errors

## Project Structure

```
DIC2/
├── analysis.py          # Core analysis & feature selection script
├── app.py               # Flask web server
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Dashboard template
└── static/
    └── style.css        # Styling
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run analysis (generates charts + result JSONs)
#    Make sure the dataset CSV path in analysis.py matches your local path
python analysis.py

# 3. Start the web server
python app.py
# → open http://127.0.0.1:5000
```

## Dataset

[50 Startups dataset](https://www.kaggle.com/datasets/farhanmd29/50-startups) — place the CSV at the path configured in `analysis.py`.

## Results Summary

| Method | Selected Features | Test RMSE | R² |
|---|---|---|---|
| LassoCV | R&D, Admin, Marketing | $8,996 | 0.900 |
| **RFECV** | **R&D, State_FL, State_NY** | **$7,946** | **0.922** |
| SFS (Forward) | R&D, Admin | $9,152 | 0.897 |
| ElasticNetCV | R&D, Admin, Marketing | $8,996 | 0.900 |
| Exhaustive FS | R&D, Admin | $9,152 | 0.897 |

RFECV achieved the lowest test RMSE by identifying that the State dummies contribute more than Administration and Marketing Spend.
