"""Pipelines and candidate models."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("bin", "passthrough", config.BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])


def candidates() -> dict:
    """Name -> unfitted pipeline. Class weights handle the real/false imbalance."""
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": Pipeline([
            ("prep", _preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           min_samples_leaf=2, n_jobs=-1, random_state=rs)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", _preprocessor()),
            ("clf", GradientBoostingClassifier(random_state=rs)),
        ]),
    }


def feature_names(pipeline: Pipeline) -> list:
    return list(pipeline.named_steps["prep"].get_feature_names_out())
