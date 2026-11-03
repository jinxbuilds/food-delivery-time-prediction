"""Model pipelines, tuning and evaluation.

Every model is a scikit-learn Pipeline: imputation and encoding are fitted on
the training data only, and the saved pipeline accepts the feature table from
``features.build_features`` directly.
"""

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42

# Traffic has a natural order, so it is encoded that way rather than alphabetically.
TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam"]
OTHER_CATEGORICAL = [c for c in CATEGORICAL_FEATURES if c != "traffic"]


def _tree_preprocessor():
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "traffic",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(categories=[TRAFFIC_ORDER], handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                ["traffic"],
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                OTHER_CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _linear_preprocessor():
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(name, **params):
    """Return an unfitted pipeline for one of the supported models."""
    if name == "Linear Regression":
        return Pipeline([("prep", _linear_preprocessor()), ("model", LinearRegression())])
    if name == "Random Forest":
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    elif name == "XGBoost":
        model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    elif name == "LightGBM":
        model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **params)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("prep", _tree_preprocessor()), ("model", model)])


# Search spaces for RandomizedSearchCV (searched on the training set only).
SEARCH_SPACES = {
    "Random Forest": {
        "model__max_depth": [10, 14, 18, 24, None],
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features": [0.5, 0.7, 1.0],
    },
    "XGBoost": {
        "model__max_depth": randint(3, 10),
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 10),
        "model__reg_lambda": loguniform(0.1, 10),
    },
    "LightGBM": {
        "model__num_leaves": randint(15, 127),
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__subsample": uniform(0.6, 0.4),
        "model__subsample_freq": [1],
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_samples": randint(10, 60),
        "model__reg_lambda": loguniform(0.1, 10),
    },
}

# Fixed settings used during the search (kept moderate so the search is quick).
BASE_PARAMS = {
    "Random Forest": {"n_estimators": 200},
    "XGBoost": {"n_estimators": 500},
    "LightGBM": {"n_estimators": 500},
}


def tune(name, X_train, y_train, n_iter=20, cv=3, verbose=0):
    """Randomized hyperparameter search with cross-validation on the training set.

    Returns (best_params, best_cv_rmse) where best_params are model kwargs.
    """
    search = RandomizedSearchCV(
        make_pipeline(name, **BASE_PARAMS[name]),
        SEARCH_SPACES[name],
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        verbose=verbose,
    )
    search.fit(X_train, y_train)
    params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    params = {**BASE_PARAMS[name], **params}
    return params, -search.best_score_


def cross_val_rmse(pipeline, X, y, cv=5):
    """Mean cross-validated RMSE, for models that are not tuned (the baseline)."""
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(
        pipeline, X, y, scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
    )
    return -scores.mean()


def evaluate(pipeline, X_test, y_test):
    """Test-set metrics for a fitted pipeline."""
    pred = pipeline.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }
