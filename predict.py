"""Predict food delivery times with the trained pipeline.

Predict for every order in a CSV that uses the Kaggle column layout:
    python predict.py --csv data/train.csv --limit 5
    python predict.py --csv new_orders.csv --output predictions.csv

Predict a single order from the command line:
    python predict.py --age 29 --rating 4.7 \\
        --restaurant 12.9141,77.6782 --destination 12.9441,77.7082 \\
        --order-time 19:30 --pickup-time 19:45 --date 2022-03-12 \\
        --weather Stormy --traffic Jam --order-type Meal --vehicle motorcycle \\
        --vehicle-condition 2 --multiple-deliveries 1 --festival No --city Metropolitian

Run python train.py first to create models/model.joblib.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from delivery_time.features import build_features, clean

MODEL_FILE = Path("models/model.joblib")

CHOICES = {
    "weather": ["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"],
    "traffic": ["Low", "Medium", "High", "Jam"],
    "order_type": ["Snack", "Meal", "Drinks", "Buffet"],
    "vehicle": ["motorcycle", "scooter", "electric_scooter", "bicycle"],
    "festival": ["Yes", "No"],
    "city": ["Urban", "Metropolitian", "Semi-Urban"],  # "Metropolitian" is spelled as in the dataset
}


def coordinates(text):
    try:
        lat, lon = (float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("use the form LAT,LON, for example 12.91,77.68")
    return lat, lon


def clock_time(text):
    try:
        return datetime.strptime(text, "%H:%M").strftime("%H:%M:%S")
    except ValueError:
        raise argparse.ArgumentTypeError("use 24-hour HH:MM, for example 19:30")


def iso_date(text):
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%d-%m-%Y")
    except ValueError:
        raise argparse.ArgumentTypeError("use YYYY-MM-DD, for example 2022-03-12")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL_FILE, help="trained pipeline (default: %(default)s)")

    batch = parser.add_argument_group("batch prediction")
    batch.add_argument("--csv", type=Path, help="CSV of orders in the Kaggle column layout")
    batch.add_argument("--output", type=Path, help="write the CSV with a predicted_minutes column added")
    batch.add_argument("--limit", type=int, help="only predict the first N rows")

    single = parser.add_argument_group("single order")
    single.add_argument("--age", type=float, help="courier age")
    single.add_argument("--rating", type=float, help="courier rating, 1 to 5")
    single.add_argument("--restaurant", type=coordinates, help="restaurant LAT,LON")
    single.add_argument("--destination", type=coordinates, help="delivery LAT,LON")
    single.add_argument("--order-time", type=clock_time, help="time the order was placed, HH:MM")
    single.add_argument("--pickup-time", type=clock_time, help="time the courier picked it up, HH:MM")
    single.add_argument("--date", type=iso_date, default=datetime.now().strftime("%Y-%m-%d"), help="order date, YYYY-MM-DD (default: today)")
    single.add_argument("--weather", choices=CHOICES["weather"])
    single.add_argument("--traffic", choices=CHOICES["traffic"])
    single.add_argument("--order-type", choices=CHOICES["order_type"], default="Meal")
    single.add_argument("--vehicle", choices=CHOICES["vehicle"], default="motorcycle")
    single.add_argument("--vehicle-condition", type=int, choices=[0, 1, 2, 3], default=1)
    single.add_argument("--multiple-deliveries", type=int, choices=[0, 1, 2, 3], default=0)
    single.add_argument("--festival", choices=CHOICES["festival"], default="No")
    single.add_argument("--city", choices=CHOICES["city"], default="Metropolitian")

    return parser, parser.parse_args()


def single_order_frame(args):
    """Build a one-row DataFrame in the raw Kaggle layout from CLI arguments."""
    return pd.DataFrame([{
        "Delivery_person_Age": args.age,
        "Delivery_person_Ratings": args.rating,
        "Restaurant_latitude": args.restaurant[0],
        "Restaurant_longitude": args.restaurant[1],
        "Delivery_location_latitude": args.destination[0],
        "Delivery_location_longitude": args.destination[1],
        "Order_Date": args.date,
        "Time_Orderd": args.order_time,
        "Time_Order_picked": args.pickup_time,
        "Weatherconditions": f"conditions {args.weather}" if args.weather else "NaN",
        "Road_traffic_density": args.traffic or "NaN",
        "Vehicle_condition": args.vehicle_condition,
        "Type_of_order": args.order_type,
        "Type_of_vehicle": args.vehicle,
        "multiple_deliveries": args.multiple_deliveries,
        "Festival": args.festival,
        "City": args.city,
    }])


def main():
    parser, args = parse_args()

    if not args.model.exists():
        sys.exit(f"Model not found at {args.model}. Run: python train.py")
    pipeline = joblib.load(args.model)

    if args.csv:
        raw = pd.read_csv(args.csv)
        if args.limit:
            raw = raw.head(args.limit)
        predictions = pipeline.predict(build_features(clean(raw))).round(1)

        if args.output:
            raw.assign(predicted_minutes=predictions).to_csv(args.output, index=False)
            print(f"Wrote {len(raw)} predictions to {args.output}")
        else:
            label = raw["ID"].astype(str).str.strip() if "ID" in raw.columns else pd.Series(range(len(raw)))
            for order_id, minutes in zip(label, predictions):
                print(f"{order_id}\t{minutes:.1f} min")
        return

    required = ["restaurant", "destination", "pickup_time"]
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"for a single order, give at least {', '.join(missing)} (or use --csv)")

    minutes = pipeline.predict(build_features(clean(single_order_frame(args))))[0]
    print(f"Predicted delivery time: {minutes:.0f} minutes")


if __name__ == "__main__":
    main()
