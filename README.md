# 🦆 Duckstradamus

> **The future price of energy.**
> A machine-learning system that forecasts New Zealand's hourly wholesale electricity
> prices for the next 24 hours — and turns that forecast into a simple decision:
> *run your heaviest load during these hours and pay less for the same consumption.*

Built by **Jacques · David · Anu** — Le Wagon Tokyo, 2026.

---

## 📌 The Problem

Wholesale electricity prices are volatile. The same air conditioning, run for the same
hours, can cost **5× more** from one night to the next — with no warning. At industrial
scale, that swing turns a ~$1K night into a ~$5K night overnight.

You didn't do anything differently. The *price* did.

**Duckstradamus** exists to make that volatility predictable and actionable.

---

## ✨ What It Does

Duckstradamus delivers a **decision, not just a forecast**:

1. **Price forecast** — tomorrow's hourly wholesale electricity prices (next 24 hours).
2. **Recommendation** — "run your maximum load between these hours tomorrow."
3. **Measurable saving** — same total consumption, same operations, lower bill.

No change to *what* you do — only *when* you do it. The timing is where the savings live.

---

## 🎯 Goal

- Predict New Zealand's wholesale electricity price.
- Serve as a **proof of concept** that generalises to larger grids worldwide.

---

## 🧠 How It Works — Data & Models

### Data

- **10 years** of **hourly** NZ electricity-market data (2014-01-01 → 2024-12-31).
- **~80 features** spanning two domains: **Energy** and **Weather**.
- **Source:** NZ Electricity Authority / Electricity Market Information (EMI).

Feature families used:

- Wholesale prices (per grid node)
- Demand by region/zone
- Generation by fuel type (Hydro, Solar, Geo, Wind, …)
- HVDC inter-island transfer
- Scheduled outages by technology
- Lake storage levels (hydro reservoirs)
- Weather across major centres (wind, solar irradiance, temperature)
- Calendar features (weekends + NZ public holidays)

### Models

Two models were tested **head-to-head**, and the strongest performer was selected:

| Model     | Type                          | Notes                                              |
|-----------|-------------------------------|----------------------------------------------------|
| **XGBoost** | Tree-based machine learning   | Strong baseline, highly interpretable              |
| **LSTM**    | Deep learning (recurrent NN)  | Designed for time series, captures sequential cycles |

Time-series structure matters here: the **order** of data, plus **daily and yearly
cycles**, drive the patterns the models learn.

---

## 📊 Results

We beat the baseline, then beat the published literature.

**Benchmark 1 — The Naive Baseline** (`tomorrow's price = today's price`)
A surprisingly hard benchmark that most published models fail to beat consistently.
> Duckstradamus is more accurate than naive on **258 / 365 days**.

**Benchmark 2 — Published Academic Work**
A peer-reviewed study on NZ electricity price forecasting reported a **1.30** error;
our XGBoost achieves **1.12**.
> **≈13.8% reduction** in forecasting error vs the published study — external
> validation, not just beating our own baselines.

On large price peaks, error is around **43 $/MWh (~12%)** — peaks remain the hardest
part to predict.

---

## 🗂️ Project Structure

```
.
├── full_cleaning_preprocessing_script.py   # End-to-end data preprocessing pipeline
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
└── data_input/                             # Raw input data (structure below)
    ├── wholesale_prices.csv
    ├── demand_by_zone.csv
    ├── hvdc_transfer.csv
    ├── scheduled_outages.csv
    ├── generation_output_merged.csv
    ├── Wind_data_100m.csv
    ├── Solar_data.csv
    ├── Temperature_data.csv
    └── Lakes storage levels/               # one CSV per lake
```

> ⚠️ Input file names are referenced directly by the pipeline — keep them exact.

---

## 🔧 Installation

```bash
# (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`** (minimum for the preprocessing pipeline):

```
pandas
holidays
tzdata
```

> Modelling (XGBoost / LSTM) additionally requires packages such as `xgboost`,
> `tensorflow`/`keras`, `scikit-learn`, `numpy`, and `matplotlib`. Add these to
> `requirements.txt` as your modelling code lands.

---

## 🚀 Usage — Preprocessing Pipeline

The preprocessing pipeline cleans every raw source, aligns it to a fixed **UTC+12**
hourly timeline, and merges everything into one master table ready for modelling.

```python
from full_cleaning_preprocessing_script import run_full_preprocessing

df_master = run_full_preprocessing(
    data_folder       = "data_input",
    start_date        = "2014-01-01",
    end_date          = "2024-12-31",
    save_path         = "preprocessed_data.csv",  # optional
    save_intermediate = False                     # True = also save per-source CSVs
)

print(df_master.shape)
df_master.head()
```

### Parameters

| Argument            | Type   | Description                                                              |
|---------------------|--------|--------------------------------------------------------------------------|
| `data_folder`       | `str`  | Folder containing all raw input files.                                   |
| `start_date`        | `str`  | Start of the target date range, e.g. `"2014-01-01"`.                     |
| `end_date`          | `str`  | End of the target date range, e.g. `"2024-12-31"`.                       |
| `save_path`         | `str`  | *(optional)* Output path for the final master CSV. `None` = don't save.  |
| `save_intermediate` | `bool` | *(optional)* If `True`, each step also writes a CSV to `data_output/`.   |

### What the pipeline does

The **wholesale-price** table is the *anchor*: it defines the full hourly grid, and
every other source is **left-joined** onto it (so gaps become `NaN`, not dropped rows).

| # | Step              | Aggregation        |
|---|-------------------|--------------------|
| 1 | Wholesale price   | hourly **mean**    |
| 2 | Generation output | hourly **sum**     |
| 3 | Wind              | clean + filter     |
| 4 | Solar             | clean + filter     |
| 5 | Temperature       | clean + filter     |
| 6 | Lake storage      | hourly interpolate |
| 7 | Holidays          | hourly flag        |
| 8 | Demand per zone   | hourly **sum**     |
| 9 | HVDC + outages    | hourly mean        |

**Key engineering details**

- **Timezone:** all sources converted from `Pacific/Auckland` to fixed `Etc/GMT-12`,
  removing daylight-saving variation.
- **DST handling:** repeated ("fall back") and missing ("spring forward") hours are
  handled explicitly per source.
- **Aggregation semantics:** rates (price, flow, temperature) are averaged; quantities
  (demand GWh, generation kWh) are summed.
- **Sparse data:** lake levels are time-interpolated to hourly rather than forward-filled.

---

## 🛣️ Where This Goes

Beyond freezers, and beyond New Zealand — any operation that can shift *when* it uses
power can save money:

- **Data centres** — schedule AI training and batch workloads around the cheapest hours.
- **Manufacturing** — sequence energy-intensive processes around renewable surplus.
- **EV fleet charging** — optimise charging for the lowest overnight prices.
- **Hydrogen producers** — run electrolysers when renewable generation drives spot prices low.

> Save a business thousands a day, millions a year — across industries, across the globe.

---

## 👤 Team

- **Jacques**
- **David**
- **Anu**

Le Wagon Tokyo · 2026

<!-- Add emails / GitHub / LinkedIn handles here if you'd like -->

---

## 📄 License

_Specify a license here (e.g. MIT) or mark as private/coursework._

## 🙏 Acknowledgements

- Data: **New Zealand Electricity Authority** (Electricity Market Information, EMI).
- Public holiday data via the [`holidays`](https://pypi.org/project/holidays/) library.
- Built during the **Le Wagon Tokyo** data science bootcamp.
