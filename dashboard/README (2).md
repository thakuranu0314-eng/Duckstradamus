# Duckstradamus — Smart Charging UI

A self-contained web UI that demonstrates the XGBoost price model for
optimal electricity charging scheduling.

There are **two versions** of the same UI:
- **Static HTML** (`index.html`) — no install, just open in a browser.
- **Streamlit** (`app.py`) — proper web app, easy to deploy/share.

## Setup

1. **Generate the data** (run once — produces both `data.js` and `data.json`):
   ```bash
   cd UI
   python generate_data.py
   ```
   Loads the saved model, predicts all of 2024, and writes predicted / actual /
   naive prices per hour for every day.

2a. **Static HTML** — just open `index.html` in any browser. Runs client-side, no server.

2b. **Streamlit** — install deps and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
   Opens at http://localhost:8501. Deploy free via Streamlit Community Cloud
   (push the UI/ folder to GitHub, point Streamlit Cloud at `app.py`).

## Pages

### Daily Charging
- Enter **daily energy demand** (MWh) and **max power capacity** (MW).
- Press **Go** — the predicted price curve for 2024-07-04 draws in hour by hour,
  then the cheapest charging hours light up one by one.
- The cost stats use the **true price** at the chosen hours (what you'd actually pay).
- Toggle **Show true prices** to overlay the actual price curve and see how
  accurate the prediction was.

### 2024 Savings
- Uses the same energy/power settings from the Daily page.
- Shows total money saved across all of 2024 by smart-charging
  (cheapest predicted hours) vs charging at a flat spread.
- Cumulative savings chart compares the model against the naive baseline.

## Files
| File | Purpose |
|---|---|
| `index.html` | Static UI (HTML + CSS + JS, Chart.js via CDN) |
| `app.py` | Streamlit UI (Plotly charts) |
| `generate_data.py` | Runs the model, exports `data.js` + `data.json` |
| `data.js` | Hourly price data for the HTML UI |
| `data.json` | Same data for the Streamlit app |
| `requirements.txt` | Python deps for the Streamlit app |

## Notes
- To use a different saved model, edit `MODEL_PATH` in `generate_data.py` and re-run it.
- To change the demo day, edit the `DEMO_DAY` selection in `generate_data.py`.
- Internet connection required on first load (Chart.js loads from CDN).
