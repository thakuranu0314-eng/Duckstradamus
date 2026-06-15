# 🦆 Duckstradamus — New Zealand Electricity Price Forecasting

> **The future price of energy.**
> A machine-learning system that forecasts New Zealand's hourly wholesale electricity
> prices for the next 24 hours — and turns that forecast into a simple decision:
> *run your heaviest load during these hours and pay less for the same consumption.*

Built by **Jacques · David · Anu** — Le Wagon Tokyo, 2026.

<!-- Add a banner / architecture image here once you have one:
![Duckstradamus](images/banner.png)
-->

---

## 📌 The Problem

Wholesale electricity prices are volatile. The same air conditioning, run for the same
hours, can cost **5× more** from one night to the next — with no warning. At industrial
scale, that swing turns a ~$1K night into a ~$5K night overnight.

You didn't do anything differently. The *price* did. **Duckstradamus** exists to make
that volatility predictable and actionable.

---

## ✨ What It Does

A **decision, not just a forecast**:

1. **Price forecast** — tomorrow's hourly wholesale electricity prices (next 24 hours).
2. **Recommendation** — "run your maximum load between these hours tomorrow."
3. **Measurable saving** — same total consumption, same operations, lower bill.

No change to *what* you do — only *when* you do it.

---

## 🧠 Data & Models

**Data:** 10 years of hourly NZ electricity-market data (2014 → 2024), ~80 features
across **Energy** and **Weather**. Source: NZ Electricity Authority (EMI).

Feature families: wholesale prices, demand by region, generation by fuel type, HVDC
inter-island transfer, scheduled outages, lake storage levels, weather (wind, solar,
temperature), and calendar features (weekends + NZ public holidays).

**Models** (tested head-to-head; XGBoost selected as the strongest):

| Model       | Type                         | Notes                                          |
|-------------|------------------------------|------------------------------------------------|
| **XGBoost** | Tree-based gradient boosting | Final model — strong & interpretable           |
| **LSTM**    | Recurrent neural network     | Captures sequential daily/yearly cycles        |
| **Prophet** | Additive time-series model   | Trend + seasonality baseline                   |

---

## 📊 Results

- Beats the **naive baseline** (`tomorrow = today`) on **258 / 365 days**.
- XGBoost error of **1.12** vs a published study's **1.30** — a **~13.8% reduction**.
- On large price peaks, error is around **43 $/MWh (~12%)** — peaks remain the hardest.

---

## 🗂️ Project Structure

```
newzealand-electricity-demand-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data_processing/        # Cleaning & preprocessing pipeline (UTC+12, hourly merge)
│   └── preprocessing.py     # run_full_preprocessing(...) — builds the master dataset
│
├── feature_engineering/    # Feature building & selection
│
├── models/                 # One folder per model
│   ├── xgboost/             # XGBoost training & evaluation (final model)
│   ├── prophet/             # Prophet baseline
│   └── lstm/                # LSTM deep-learning model
│
├── dashboard/              # Live prediction & recommendation app
│
├── notebooks/              # Exploratory & analysis notebooks
│
└── images/                 # Figures, plots & diagrams used in the README / reports
```

> ⚠️ Raw and processed **data** (CSV / XLSX) and `__pycache__` are **git-ignored** —
> they are never committed. Keep your raw input data locally (e.g. in a `data_input/`
> folder) and point the pipeline at it. Only **figures** in `images/` are committed.

---

## 🔧 Installation

```bash
# Clone
git clone https://github.com/<your-username>/newzealand-electricity-demand-forecasting.git
cd newzealand-electricity-demand-forecasting

# (Recommended) virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage — Preprocessing

The pipeline in `data_processing/preprocessing.py` cleans every raw source, converts NZ
local time to a fixed **UTC+12** offset, aggregates to hourly, and merges everything
into one master table (anchored on the wholesale-price timeline).

```python
from data_processing.preprocessing import run_full_preprocessing

df_master = run_full_preprocessing(
    data_folder       = "data_input",   # your local raw-data folder (git-ignored)
    start_date        = "2014-01-01",
    end_date          = "2024-12-31",
    save_path         = "preprocessed_data.csv",   # optional
    save_intermediate = False,
)
```

**Expected raw-data layout** (inside `data_input/`):

```
data_input/
├── wholesale_prices.csv
├── demand_by_zone.csv
├── hvdc_transfer.csv
├── scheduled_outages.csv
├── generation_output_merged.csv
├── Wind_data_100m.csv
├── Solar_data.csv
├── Temperature_data.csv
└── Lakes storage levels/      # one CSV per lake
```

**Key engineering details:** all sources converted from `Pacific/Auckland` to fixed
`Etc/GMT-12` (removes DST); rates (price, flow, temperature) are averaged while
quantities (demand GWh, generation kWh) are summed; sparse lake levels are
time-interpolated to hourly.

---

## 🔁 Pipeline Order

1. **Data processing** — `data_processing/preprocessing.py` → master hourly dataset.
2. **Feature engineering** — `feature_engineering/` → model-ready features.
3. **Modelling** — `models/{xgboost,prophet,lstm}/` → train & evaluate.
4. **Dashboard** — `dashboard/` → live prediction & load-shifting recommendation.

---

## 🛣️ Where This Goes

Beyond New Zealand — any operation that can shift *when* it uses power: **data centres**
(schedule AI/batch jobs), **manufacturing** (sequence around renewable surplus), **EV
fleet charging** (cheapest overnight hours), and **hydrogen producers** (run
electrolysers when spot prices are low).

---

## 👤 Team

- **Jacques**
- **David**
- **Anu**

Le Wagon Tokyo · 2026

## 📄 License

_Specify a license here (e.g. MIT) or mark as coursework._

## 🙏 Acknowledgements

- Data: **New Zealand Electricity Authority** (Electricity Market Information, EMI).
- Public holidays via the [`holidays`](https://pypi.org/project/holidays/) library.
- Built during the **Le Wagon Tokyo** data science bootcamp.
