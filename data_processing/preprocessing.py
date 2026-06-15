# ==========================================
# DUCKSTRADAMUS — CONSOLIDATED PREPROCESSING PIPELINE
# ==========================================
#
# PURPOSE:
#   Single-file version of all cleaning and preprocessing logic.
#   Replaces the four separate modules (clean_emi_data.py,
#   preprocess_emi_data.py, clean_generation_output_data.py,
#   preprocessing_Anu.py) and the main_preprocess.ipynb orchestration.
#
# HOW TO USE:
#   from duckstradamus.data.preprocessing import run_full_preprocessing
#
#   df_master = run_full_preprocessing(
#       data_folder       = "data/raw",
#       start_date        = "2014-01-01",
#       end_date          = "2024-12-31",
#       save_path         = "data/processed/preprocessed_data.csv",  # optional
#       save_intermediate = False   # set True to also save per-source CSVs
#   )
#
# EXPECTED data_folder STRUCTURE:
#   data_folder/
#     wholesale_prices.csv
#     demand_by_zone.csv
#     hvdc_transfer.csv
#     scheduled_outages.csv
#     generation_output_merged.csv
#     Wind_data_100m.csv
#     Solar_data.csv
#     Temperature_data.csv
#     Lakes storage levels/     <- subfolder, one CSV per lake
# ==========================================

import os
import pandas as pd
import holidays


# ==========================================
# SECTION 1 — WHOLESALE PRICE
# ==========================================

def clean_wholesale_price(data_folder, save_intermediate=False):
    """
    Loads raw wholesale price CSV, unpivots region nodes into columns,
    and converts timestamps from NZ local time (Pacific/Auckland) to
    UTC+12 fixed offset (Etc/GMT-12), removing daylight saving variation.
    """
    file_path = os.path.join(data_folder, "wholesale_prices.csv")
    wholesale_price = pd.read_csv(file_path)

    wholesale_price = wholesale_price[
        ['Period start', 'Region ID', 'Price ($/MWh)']
    ].rename(columns={
        'Period start':   'datetime',
        'Region ID':      'region_id',
        'Price ($/MWh)':  'price_dol/MWh'
    })

    wholesale_price['datetime'] = pd.to_datetime(
        wholesale_price['datetime'], dayfirst=True
    )

    wholesale_price = wholesale_price.pivot(
        index='datetime', columns='region_id', values='price_dol/MWh'
    )
    wholesale_price.columns = [
        f'el_price_dol/MWh_{col}' for col in wholesale_price.columns
    ]

    # --- DST handling ---
    extra_hour_data       = wholesale_price[wholesale_price.index.minute.isin([40, 50])].copy()
    wholesale_price_clean = wholesale_price[~wholesale_price.index.minute.isin([40, 50])].copy()

    wholesale_price_utc12 = wholesale_price_clean.copy()
    wholesale_price_utc12.index = (
        pd.to_datetime(wholesale_price_utc12.index)
        .tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )
    wholesale_price_utc12 = wholesale_price_utc12[wholesale_price_utc12.index.notna()]

    extra_hour_data.index = extra_hour_data.index.map(
        lambda ts: ts.replace(minute=0 if ts.minute == 40 else 30)
    )

    wholesale_price_utc12 = pd.concat(
        [wholesale_price_utc12, extra_hour_data]
    ).sort_index()

    dupes = wholesale_price_utc12[wholesale_price_utc12.index.duplicated(keep=False)]
    if not dupes.empty:
        print(f"WARNING: Duplicate timestamps in wholesale price after conversion:\n{dupes}")

    wholesale_price_utc12.reset_index(inplace=True)

    if save_intermediate:
        out = "data_output/wholesale_price_utc12.csv"
        wholesale_price_utc12.to_csv(out, index=False)
        print(f"   saved intermediate -> {out}")

    return wholesale_price_utc12


def preprocess_wholesale_price(df, start_date=None, end_date=None, save_intermediate=False):
    """Aggregates 30-min wholesale price data to hourly buckets using mean."""
    df['hour_bucket'] = (
        df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly = (
        df
        .groupby('hour_bucket')
        .agg(
            datetime_utc12            = ('datetime',                  'max'),
            el_price_dol_MWh_BEN2201  = ('el_price_dol/MWh_BEN2201',  'mean'),
            el_price_dol_MWh_HAY2201  = ('el_price_dol/MWh_HAY2201',  'mean'),
            el_price_dol_MWh_INV2201  = ('el_price_dol/MWh_INV2201',  'mean'),
            el_price_dol_MWh_ISL2201  = ('el_price_dol/MWh_ISL2201',  'mean'),
            el_price_dol_MWh_KIK2201  = ('el_price_dol/MWh_KIK2201',  'mean'),
            el_price_dol_MWh_OTA2201  = ('el_price_dol/MWh_OTA2201',  'mean'),
            el_price_dol_MWh_RDF2201  = ('el_price_dol/MWh_RDF2201',  'mean'),
            el_price_dol_MWh_SFD2201  = ('el_price_dol/MWh_SFD2201',  'mean'),
            el_price_dol_MWh_WKM2201  = ('el_price_dol/MWh_WKM2201',  'mean'),
        )
        .reset_index(drop=True)
        .sort_values('datetime_utc12')
        .reset_index(drop=True)
    )

    if start_date is not None:
        df_hourly = df_hourly[df_hourly['datetime_utc12'] >= start_date]
    if end_date is not None:
        df_hourly = df_hourly[df_hourly['datetime_utc12'] <= end_date]
    df_hourly = df_hourly.reset_index(drop=True)

    if save_intermediate:
        out = "data_output/wholesale_price_preprocessed.csv"
        df_hourly.to_csv(out, index=False)
        print(f"   saved intermediate -> {out}")

    return df_hourly


# ==========================================
# SECTION 2 — DEMAND PER ZONE
# ==========================================

def clean_demand_per_zone(data_folder, save_intermediate=False):
    """Loads raw demand CSV, unpivots zone columns, and converts to UTC+12."""
    file_path = os.path.join(data_folder, "demand_by_zone.csv")
    demand = pd.read_csv(file_path)

    demand = demand[
        ['Period start', 'Region ID', 'Demand (GWh)']
    ].rename(columns={
        'Period start':  'datetime',
        'Region ID':     'region_id',
        'Demand (GWh)':  'demand_GWh'
    })
    demand['datetime'] = pd.to_datetime(demand['datetime'], dayfirst=True)

    demand = demand.pivot(index='datetime', columns='region_id', values='demand_GWh')
    demand.columns = [f'demand_GWh_{col}' for col in demand.columns]

    # --- DST handling ---
    extra_hour_data = demand[demand.index.minute.isin([40, 50])].copy()
    demand_clean    = demand[~demand.index.minute.isin([40, 50])].copy()

    demand_utc12 = demand_clean.copy()
    demand_utc12.index = (
        pd.to_datetime(demand_utc12.index)
        .tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )
    demand_utc12 = demand_utc12[demand_utc12.index.notna()]

    extra_hour_data.index = extra_hour_data.index.map(
        lambda ts: ts.replace(minute=0 if ts.minute == 40 else 30)
    )
    demand_utc12 = pd.concat([demand_utc12, extra_hour_data]).sort_index()

    dupes = demand_utc12[demand_utc12.index.duplicated(keep=False)]
    if not dupes.empty:
        print(f"WARNING: Duplicate timestamps in demand after conversion:\n{dupes}")

    demand_utc12.reset_index(inplace=True)

    if save_intermediate:
        out = "data_output/demand_utc12.csv"
        demand_utc12.to_csv(out, index=False)
        print(f"   saved intermediate -> {out}")

    return demand_utc12


def preprocess_demand_per_zone(df, save_intermediate=False):
    """Aggregates 30-min demand data to hourly buckets using sum."""
    df['hour_bucket'] = (
        df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_demand = (
        df
        .groupby('hour_bucket')
        .agg(
            datetime_utc12  = ('datetime',       'max'),
            demand_GWh_CNI  = ('demand_GWh_CNI', 'sum'),
            demand_GWh_LNI  = ('demand_GWh_LNI', 'sum'),
            demand_GWh_LSI  = ('demand_GWh_LSI', 'sum'),
            demand_GWh_UNI  = ('demand_GWh_UNI', 'sum'),
            demand_GWh_USI  = ('demand_GWh_USI', 'sum'),
        )
        .reset_index(drop=True)
        .sort_values('datetime_utc12')
        .reset_index(drop=True)
    )

    if save_intermediate:
        out = "data_output/demand_preprocessed.csv"
        df_hourly_demand.to_csv(out, index=False)
        print(f"   saved intermediate -> {out}")

    return df_hourly_demand


# ==========================================
# SECTION 3 — HVDC TRANSFER
# ==========================================

def clean_hvdc(data_folder, save_intermediate=False):
    """Loads HVDC transfer data, handles duplicates, converts to UTC+12, fills gaps."""
    file_path = os.path.join(data_folder, "hvdc_transfer.csv")
    hvdc = pd.read_csv(file_path)

    hvdc = hvdc[
        ['Period start', 'Direction', 'Peak flow (MW)', 'Average flow (MW)']
    ].rename(columns={
        'Period start':       'datetime',
        'Peak flow (MW)':     'peak_flow_MW',
        'Average flow (MW)':  'avg_flow_MW'
    })
    hvdc['datetime'] = pd.to_datetime(hvdc['datetime'], dayfirst=True)

    hvdc_dupes = hvdc[hvdc['datetime'].duplicated(keep='first')].copy()
    hvdc_clean = hvdc[~hvdc['datetime'].duplicated(keep='first')].copy()

    hvdc_clean = hvdc_clean.set_index('datetime')
    hvdc_clean.index = (
        pd.to_datetime(hvdc_clean.index)
        .tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )
    hvdc_clean = hvdc_clean[hvdc_clean.index.notna()]

    hvdc_dupes = hvdc_dupes.set_index('datetime')
    hvdc_dupes.index = (
        pd.to_datetime(hvdc_dupes.index)
        .tz_localize('Pacific/Auckland', ambiguous=True, nonexistent='NaT')
        .tz_convert('Etc/GMT-12')
        .tz_localize(None)
    )

    new_slots = hvdc_dupes.index[~hvdc_dupes.index.isin(hvdc_clean.index)]
    hvdc_clean = pd.concat([hvdc_clean, hvdc_dupes.loc[new_slots]]).sort_index()

    hvdc_utc12 = hvdc_clean.resample('30min').asfreq().fillna(0)
    hvdc_utc12.reset_index(inplace=True)

    if save_intermediate:
        out = "data_output/hvdc_utc12.csv"
        hvdc_utc12.to_csv(out, index=False)
        print(f"   saved intermediate -> {out}")

    return hvdc_utc12


def preprocess_hvdc(df, save_intermediate=False):
    """Aggregates 30-min HVDC data to hourly buckets, derives a direction indicator."""
    df['hour_bucket'] = (
        df['datetime'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_hvdc = (
        df
        .groupby('hour_bucket')
        .agg(
            datetime_utc12  = ('datetime',      'max'),
            avg_flow_MW     = ('avg_flow_MW',   'mean'),
            peak_flow_MW    = ('peak_flow_MW',  lambda x: x.loc[x.abs().idxmax()]),
        )
        .reset_index(drop=True)
        .sort_values('datetime_utc12')
        .reset_index(drop=True)
    )

    df_hourly_hvdc['Direction'] = df_hourly_hvdc['avg_flow_MW'].map(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    if save_intermediate:
        out = "data_output/hvdc_preprocessed.csv"
        df_hourly_hvdc.to_csv(out, index=False)
        print(f"   saved intermediate -> {out}")

    return df_hourly_hvdc


# ==========================================
# SECTION 4 — SCHEDULED OUTAGES
# ==========================================

def clean_outages(data_folder, save_intermediate=False):
    """Loads scheduled outage data, pivots by technology, builds 30-min grid, ffills."""
    file_path = os.path.join(data_folder, "scheduled_outages.csv")
    df = pd.read_csv(file_path)
    df.columns = ['Timestamp', 'SeriesCode', 'Series', 'MW']

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)

    df['Timestamp'] = (
        df['Timestamp']
        .dt.tz_localize('Pacific/Auckland', ambiguous='NaT', nonexistent='NaT')
        .dt.tz_convert('Etc/GMT-12')
        .dt.tz_localize(None)
    )
    df = df[df['Timestamp'].notna()]

    df_wide = df.pivot_table(
        index='Timestamp', columns='SeriesCode', values='MW', aggfunc='first'
    )
    df_wide.columns.name = None

    full_index = pd.date_range(
        start=df_wide.index.min(),
        end=df_wide.index.max(),
        freq='30min'
    )
    df_outages = df_wide.reindex(full_index).ffill().fillna(0)
    df_outages.index.name = 'Timestamp'
    df_outages.reset_index(inplace=True)

    if save_intermediate:
        out = "data_output/scheduled_outages_utc12.csv"
        df_outages.to_csv(out, index=False)
        print(f"   saved intermediate -> {out}")

    return df_outages


def preprocess_outages(df, save_intermediate=False):
    """Aggregates 30-min outage data to hourly buckets using mean MW per technology."""
    df['hour_bucket'] = (
        df['Timestamp'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly_outages = (
        df
        .groupby('hour_bucket')
        .agg(
            datetime_utc12  = ('Timestamp', 'max'),
            outage_Gas_MW   = ('NZ_G',      'mean'),
            outage_Hyd_MW   = ('NZ_H',      'mean'),
            outage_Ter_MW   = ('NZ_T',      'mean'),
            outage_Win_MW   = ('NZ_W',      'mean'),
            outage_UNKN_MW  = ('UNKN',      'mean'),
        )
        .reset_index(drop=True)
        .sort_values('datetime_utc12')
        .reset_index(drop=True)
    )

    if save_intermediate:
        out = "data_output/outages_preprocessed.csv"
        df_hourly_outages.to_csv(out, index=False)
        print(f"   saved intermediate -> {out}")

    return df_hourly_outages


# ==========================================
# SECTION 5 — GENERATION OUTPUT
# ==========================================

def preprocess_generation_data(file_path, start_date, end_date):
    """Full cleaning + preprocessing pipeline for generation output data."""
    df = pd.read_csv(file_path, low_memory=False)

    # --- Fix dual-schema Trading_Date column ---
    df['Trading_date'] = df['Trading_date'].fillna(df['Trading_Date'])
    df = df.drop(columns=['Trading_Date'])
    df = df.rename(columns={'Trading_date': 'Trading_Date'})

    # --- Melt: wide -> long ---
    metadata_cols = [
        'Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
        'Fuel_Code', 'Tech_Code', 'Trading_Date'
    ]
    tp_cols = [f'TP{i}' for i in range(1, 51)]

    df_long = df.melt(
        id_vars=metadata_cols,
        value_vars=tp_cols,
        var_name='TP',
        value_name='generation_kwh'
    )

    df_long = df_long.dropna(subset=['generation_kwh'])

    df_long['tp_num'] = df_long['TP'].str.replace('TP', '').astype(int)
    df_long = df_long.drop(columns=['TP'])

    # --- Datetime conversion ---
    df_long['datetime_nz'] = (
        pd.to_datetime(df_long['Trading_Date'])
        + pd.to_timedelta((df_long['tp_num'] - 1) * 30, unit='m')
    )

    nz_tz_aware = df_long['datetime_nz'].dt.tz_localize(
        'Pacific/Auckland',
        nonexistent=pd.Timedelta(hours=1),
        ambiguous=True
    )

    df_long['datetime_utc']   = nz_tz_aware.dt.tz_convert('UTC').dt.tz_localize(None)
    df_long['datetime_utc12'] = nz_tz_aware.dt.tz_convert('Etc/GMT-12').dt.tz_localize(None)

    df_long = df_long.drop(columns=['Trading_Date', 'tp_num'])

    final_order = [
        'datetime_nz', 'datetime_utc12', 'datetime_utc',
        'Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
        'Fuel_Code', 'Tech_Code', 'generation_kwh',
    ]
    df_long = (
        df_long[final_order]
        .sort_values(['datetime_utc12', 'POC_Code'])
        .reset_index(drop=True)
    )

    # --- Aggregate 30-min -> hourly ---
    df_long['hour_bucket'] = (
        df_long['datetime_utc12'] + pd.Timedelta(minutes=30)
    ).dt.floor('h')

    df_hourly = (
        df_long
        .groupby(
            ['hour_bucket', 'Site_Code', 'POC_Code', 'Nwk_Code',
             'Gen_Code', 'Fuel_Code', 'Tech_Code'],
            observed=True
        )
        .agg(
            datetime_nz    = ('datetime_nz',    'max'),
            datetime_utc12 = ('datetime_utc12', 'max'),
            datetime_utc   = ('datetime_utc',   'max'),
            generation_kwh = ('generation_kwh', 'sum'),
        )
        .reset_index()
        .drop(columns=['hour_bucket'])
    )

    final_cols = [
        'datetime_nz', 'datetime_utc12', 'datetime_utc',
        'Site_Code', 'POC_Code', 'Nwk_Code', 'Gen_Code',
        'Fuel_Code', 'Tech_Code', 'generation_kwh',
    ]
    df_hourly = (
        df_hourly[final_cols]
        .sort_values(['datetime_utc12', 'POC_Code'])
        .reset_index(drop=True)
    )

    df_hourly = df_hourly.drop(columns=['datetime_nz', 'datetime_utc'])

    # --- Standardise Fuel_Code labels ---
    fuel_mapping = {
        'HYD': 'Hydro',
        'SOL': 'Solar',
        'GEO': 'Geo',
        'WIN': 'Wind',
        'ELE': 'Ele',
    }
    df_hourly['Fuel_Code'] = df_hourly['Fuel_Code'].replace(fuel_mapping)

    # --- Pivot by Fuel_Code ---
    df_by_fuel = df_hourly.pivot_table(
        index='datetime_utc12',
        columns='Fuel_Code',
        values='generation_kwh',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    df_by_fuel.columns.name = None

    df_by_fuel = df_by_fuel[
        (df_by_fuel['datetime_utc12'] >= start_date) &
        (df_by_fuel['datetime_utc12'] <= end_date)
    ].reset_index(drop=True)

    return df_by_fuel


# ==========================================
# SECTIONS 6-8 — WIND / SOLAR / TEMPERATURE
# ==========================================

def _clean_weather_like(file_path, start_date=None, end_date=None):
    """Shared cleaning logic for wind, solar and temperature files."""
    df = pd.read_csv(file_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df.drop(columns=["time"], inplace=True)
    df = df[["datetime"] + [col for col in df.columns if col != "datetime"]]

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.sort_values("datetime")

    if start_date is not None:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def preprocess_wind_data(file_path, start_date=None, end_date=None):
    """Loads and cleans wind speed/direction data."""
    return _clean_weather_like(file_path, start_date, end_date)


def preprocess_solar_data(file_path, start_date=None, end_date=None):
    """Loads and cleans solar irradiance data."""
    return _clean_weather_like(file_path, start_date, end_date)


def preprocess_temperature_data(file_path, start_date=None, end_date=None):
    """Loads and cleans temperature data."""
    return _clean_weather_like(file_path, start_date, end_date)


# ==========================================
# SECTION 9 — LAKE STORAGE
# ==========================================

def preprocess_lake_storage(file_path):
    """Loads a single lake storage CSV and returns key columns with a clean datetime."""
    df = pd.read_csv(file_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        errors="coerce"
    )

    df = df[["datetime", "lake_level_m", "active_storage_mm³"]]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def make_lake_hourly(df, start_date=None, end_date=None):
    """Resamples a single lake storage DataFrame to hourly and interpolates (time-weighted)."""
    df["datetime"] = pd.to_datetime(df["datetime"])

    mask = df["datetime"].dt.time == pd.to_datetime("23:59:59").time()
    df.loc[mask, "datetime"] = df.loc[mask, "datetime"] + pd.Timedelta(seconds=1)

    df = df.sort_values("datetime")

    if start_date is not None:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    df = df.set_index("datetime")
    hourly_df = df.resample("h").asfreq()
    hourly_df = hourly_df.infer_objects(copy=False)

    numeric_cols = hourly_df.select_dtypes(include="number").columns
    hourly_df[numeric_cols] = hourly_df[numeric_cols].interpolate(method="time")

    return hourly_df.reset_index()


# ==========================================
# SECTION 10 — NZ HOLIDAY DATA
# ==========================================

def create_nz_holiday_data(start_date, end_date):
    """Hourly DataFrame with a binary 'holiday' column (1 on weekends + NZ public holidays)."""
    df = pd.DataFrame({
        "datetime": pd.date_range(start=start_date, end=end_date, freq="h")
    })

    nz_holidays = holidays.NewZealand(
        years=range(
            pd.to_datetime(start_date).year,
            pd.to_datetime(end_date).year + 1
        )
    )

    is_holiday = df["datetime"].dt.date.isin(nz_holidays)
    is_weekend = df["datetime"].dt.weekday >= 5

    df["holiday"] = (is_holiday | is_weekend).astype(int)

    return df


# ==========================================
# MASTER ORCHESTRATOR
# ==========================================

def run_full_preprocessing(
    data_folder,
    start_date,
    end_date,
    save_path=None,
    save_intermediate=False
):
    """Runs the complete Duckstradamus preprocessing pipeline end-to-end."""
    print("=" * 60)
    print("DUCKSTRADAMUS — FULL PREPROCESSING PIPELINE")
    print(f"  data_folder : {data_folder}")
    print(f"  date range  : {start_date}  ->  {end_date}")
    print(f"  save_path   : {save_path}")
    print("=" * 60)

    # STEP 1 — Wholesale price (anchor)
    print("\n[1/9] Wholesale price...")
    df = preprocess_wholesale_price(
        clean_wholesale_price(data_folder, save_intermediate=save_intermediate),
        start_date=start_date,
        end_date=end_date,
        save_intermediate=save_intermediate
    )
    print(f"      Anchor df shape: {df.shape}")

    # STEP 2 — Generation output
    print("\n[2/9] Generation output...")
    df_gen = preprocess_generation_data(
        file_path=os.path.join(data_folder, "generation_output_merged.csv"),
        start_date=start_date,
        end_date=end_date
    )
    df = df.merge(df_gen, how='left', on='datetime_utc12')
    print(f"      Shape after merge: {df.shape}")

    # STEP 3 — Wind
    print("\n[3/9] Wind...")
    df_wind = preprocess_wind_data(
        file_path=os.path.join(data_folder, "Wind_data_100m.csv"),
        start_date=start_date,
        end_date=end_date
    )
    df = df.merge(df_wind, how='left', left_on='datetime_utc12', right_on='datetime')
    df = df.drop(columns=['datetime'])
    print(f"      Shape after merge: {df.shape}")

    # STEP 4 — Solar
    print("\n[4/9] Solar...")
    df_solar = preprocess_solar_data(
        file_path=os.path.join(data_folder, "Solar_data.csv"),
        start_date=start_date,
        end_date=end_date
    )
    df = df.merge(df_solar, how='left', left_on='datetime_utc12', right_on='datetime')
    df = df.drop(columns=['datetime'])
    print(f"      Shape after merge: {df.shape}")

    # STEP 5 — Temperature
    print("\n[5/9] Temperature...")
    df_temp = preprocess_temperature_data(
        file_path=os.path.join(data_folder, "Temperature_data.csv"),
        start_date=start_date,
        end_date=end_date
    )
    df = df.merge(df_temp, how='left', left_on='datetime_utc12', right_on='datetime')
    df = df.drop(columns=['datetime'])
    print(f"      Shape after merge: {df.shape}")

    # STEP 6 — Lake storage (loop over all lake files)
    print("\n[6/9] Lake storage...")
    lake_folder = os.path.join(data_folder, "Lakes storage levels")
    for filename in sorted(os.listdir(lake_folder)):
        filepath = os.path.join(lake_folder, filename)
        prefix = filename[:7]

        df_lake = make_lake_hourly(
            preprocess_lake_storage(filepath),
            start_date=start_date,
            end_date=end_date
        )
        df_lake = df_lake.rename(columns={
            col: f'{prefix}_{col}'
            for col in df_lake.columns
            if col != 'datetime'
        })
        df = df.merge(df_lake, how='left', left_on='datetime_utc12', right_on='datetime')
        df = df.drop(columns=['datetime'])
        print(f"      Merged: {filename}  (prefix: '{prefix}')")

    print(f"      Shape after all lake merges: {df.shape}")

    # STEP 7 — Holidays
    print("\n[7/9] Holidays...")
    df_holiday = create_nz_holiday_data(start_date, end_date)
    df = df.merge(df_holiday, how='left', left_on='datetime_utc12', right_on='datetime')
    df = df.drop(columns=['datetime'])
    print(f"      Shape after merge: {df.shape}")

    # STEP 8 — Demand per zone
    print("\n[8/9] Demand per zone...")
    df_demand = preprocess_demand_per_zone(
        clean_demand_per_zone(data_folder, save_intermediate=save_intermediate),
        save_intermediate=save_intermediate
    )
    df = df.merge(df_demand, how='left', on='datetime_utc12')
    print(f"      Shape after merge: {df.shape}")

    # STEP 9 — HVDC + Scheduled outages
    print("\n[9/9] HVDC and scheduled outages...")
    df_hvdc = preprocess_hvdc(
        clean_hvdc(data_folder, save_intermediate=save_intermediate),
        save_intermediate=save_intermediate
    )
    df = df.merge(df_hvdc, how='left', on='datetime_utc12')

    df_outages = preprocess_outages(
        clean_outages(data_folder, save_intermediate=save_intermediate),
        save_intermediate=save_intermediate
    )
    df = df.merge(df_outages, how='left', on='datetime_utc12')
    print(f"      Shape after merge: {df.shape}")

    # SAVE
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"   Saved to    : {save_path}")

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete.")
    print(f"   Final shape : {df.shape}")
    print(f"{'=' * 60}\n")

    return df


if __name__ == "__main__":
    run_full_preprocessing(
        data_folder="data/raw",
        start_date="2014-01-01",
        end_date="2024-12-31",
        save_path="data/processed/preprocessed_data.csv",
    )
