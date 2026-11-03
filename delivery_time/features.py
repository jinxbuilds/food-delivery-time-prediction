"""Cleaning and feature engineering for the Kaggle food delivery dataset.

Everything here works row by row and learns nothing from the data, so it is
safe to apply before the train/test split. Anything that is fitted (imputation,
encoding, scaling) lives in the scikit-learn pipeline in ``modeling.py`` so it
only ever sees training data.
"""

import numpy as np
import pandas as pd

TARGET = "delivery_time_min"

NUMERIC_FEATURES = [
    "courier_age",
    "courier_rating",
    "distance_km",
    "order_hour",
    "prep_min",
    "vehicle_condition",
    "multiple_deliveries",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "weather",
    "traffic",
    "order_type",
    "vehicle_type",
    "festival",
    "city",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw Kaggle columns needed to build the features (target optional).
RAW_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
]


def _to_number(series):
    return pd.to_numeric(series, errors="coerce")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two arrays of coordinates."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, turn the dataset's "NaN" strings into real NaN, fix types."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    df = raw.copy()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().replace({"NaN": pd.NA, "": pd.NA})

    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)
    df.loc[df["Weatherconditions"] == "NaN", "Weatherconditions"] = pd.NA

    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries", "Vehicle_condition"]:
        df[col] = _to_number(df[col])

    # Ratings are on a 1-5 scale; 53 rows have a rating of 6, which is invalid.
    df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = np.nan

    if "Time_taken(min)" in df.columns:
        df[TARGET] = _to_number(df["Time_taken(min)"].str.replace("(min)", "", regex=False))

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn cleaned raw columns into the model's feature table."""
    out = pd.DataFrame(index=df.index)

    out["courier_age"] = df["Delivery_person_Age"]
    out["courier_rating"] = df["Delivery_person_Ratings"]

    # Some coordinates have a flipped sign (all orders are in India, so every
    # value should be positive); abs() repairs them.
    out["distance_km"] = haversine_km(
        df["Restaurant_latitude"].abs(),
        df["Restaurant_longitude"].abs(),
        df["Delivery_location_latitude"].abs(),
        df["Delivery_location_longitude"].abs(),
    )

    order_date = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    ordered = pd.to_datetime(df["Order_Date"] + " " + df["Time_Orderd"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    picked = pd.to_datetime(df["Order_Date"] + " " + df["Time_Order_picked"], format="%d-%m-%Y %H:%M:%S", errors="coerce")

    # Pickup earlier than the order time means the pickup happened after midnight.
    prep = (picked - ordered).dt.total_seconds() / 60
    prep = prep.where(prep >= 0, prep + 24 * 60)
    out["prep_min"] = prep

    # When the order time is missing, the pickup hour is a close stand-in.
    out["order_hour"] = ordered.dt.hour.fillna(picked.dt.hour)
    out["is_weekend"] = (order_date.dt.dayofweek >= 5).astype(float).where(order_date.notna())

    out["vehicle_condition"] = df["Vehicle_condition"]
    out["multiple_deliveries"] = df["multiple_deliveries"]

    out["weather"] = df["Weatherconditions"]
    out["traffic"] = df["Road_traffic_density"]
    out["order_type"] = df["Type_of_order"]
    out["vehicle_type"] = df["Type_of_vehicle"]
    out["festival"] = df["Festival"]
    out["city"] = df["City"]

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    for col in NUMERIC_FEATURES:
        out[col] = out[col].astype(float)

    return out[FEATURES]


def load_dataset(path="data/train.csv"):
    """Read the Kaggle CSV and return (X, y)."""
    df = clean(pd.read_csv(path))
    df = df[df[TARGET].notna()]
    return build_features(df), df[TARGET].astype(float)
