"""Train and compare the models, then save the best pipeline.

Usage:
    python train.py            # train with the tuned parameters in params.json
    python train.py --tune     # re-run the hyperparameter search first (slower)
"""

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from delivery_time.features import load_dataset
from delivery_time.modeling import RANDOM_STATE, cross_val_rmse, evaluate, make_pipeline, tune

PARAMS_FILE = Path("params.json")
RESULTS_FILE = Path("results.json")
MODEL_FILE = Path("models/model.joblib")

TUNED_MODELS = ["Random Forest", "XGBoost", "LightGBM"]
SEARCH_ITERATIONS = {"Random Forest": 6, "XGBoost": 20, "LightGBM": 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train.csv", help="path to the Kaggle train.csv")
    parser.add_argument("--tune", action="store_true", help="re-run the hyperparameter search and update params.json")
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"{len(X_train)} training rows, {len(X_test)} test rows")

    cv_rmse = {"Linear Regression": cross_val_rmse(make_pipeline("Linear Regression"), X_train, y_train)}

    if args.tune or not PARAMS_FILE.exists():
        params = {}
        for name in TUNED_MODELS:
            start = time.time()
            params[name], cv_rmse[name] = tune(name, X_train, y_train, n_iter=SEARCH_ITERATIONS[name])
            print(f"Tuned {name} in {time.time() - start:.0f}s: CV RMSE {cv_rmse[name]:.3f}")
        PARAMS_FILE.write_text(json.dumps({"params": params, "cv_rmse": cv_rmse}, indent=2, default=float))
    else:
        saved = json.loads(PARAMS_FILE.read_text())
        params, cv_rmse = saved["params"], {**saved["cv_rmse"], **cv_rmse}

    results, fitted = {}, {}
    for name in ["Linear Regression"] + TUNED_MODELS:
        pipeline = make_pipeline(name, **params.get(name, {}))
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        results[name] = {"cv_rmse": cv_rmse[name], **evaluate(pipeline, X_test, y_test)}

    # Choose the best model by cross-validated RMSE, never by test score.
    best = min(TUNED_MODELS, key=lambda name: cv_rmse[name])

    print(f"\n{'Model':<20}{'CV RMSE':>9}{'Test RMSE':>11}{'Test MAE':>10}{'Test R²':>9}")
    for name, r in results.items():
        flag = "  <- selected" if name == best else ""
        print(f"{name:<20}{r['cv_rmse']:>9.2f}{r['rmse']:>11.2f}{r['mae']:>10.2f}{r['r2']:>9.3f}{flag}")

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_FILE, compress=3)
    RESULTS_FILE.write_text(json.dumps({"selected": best, "results": results}, indent=2))
    print(f"\nSaved {best} pipeline to {MODEL_FILE}")


if __name__ == "__main__":
    main()
