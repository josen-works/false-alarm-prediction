"""Train, compare, evaluate and save the best model."""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config
from .model import candidates, feature_names


def best_threshold(y_true, proba, min_recall: float = 0.90) -> float:
    """Highest-precision threshold that still catches >= min_recall of real alarms.

    Missing a real alarm is far costlier than a false alarm, so recall is the constraint.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= min_recall)[0]
    if len(ok) == 0:
        return 0.5
    return float(thresholds[ok[np.argmax(precision[:-1][ok])]])


def train(df: pd.DataFrame, model_path=config.MODEL_PATH, report: bool = True) -> dict:
    X, y = df[config.ALL_FEATURES], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)

    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = {}
    for name, pipe in candidates().items():
        cv_scores[name] = float(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1").mean())
        print(f"  {name:22s} CV F1 = {cv_scores[name]:.3f}")
    best_name = max(cv_scores, key=cv_scores.get)
    print(f"Best model: {best_name}")

    pipe = candidates()[best_name].fit(X_tr, y_tr)

    # Pick the threshold on training predictions via CV so the test set stays untouched.
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(candidates()[best_name], X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_tr, oof)

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_te, pred)
    metrics = {
        "best_model": best_name,
        "cv_f1": cv_scores,
        "threshold": threshold,
        "test_f1": float(f1_score(y_te, pred)),
        "test_roc_auc": float(roc_auc_score(y_te, proba)),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "classification_report": classification_report(
            y_te, pred, target_names=["false_alarm", "real_alarm"], output_dict=True),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold, "features": config.ALL_FEATURES}, model_path)

    if report:
        config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        ConfusionMatrixDisplay(cm, display_labels=["false", "real"]).plot(cmap="Blues")
        plt.title(f"{best_name} (threshold {threshold:.2f})")
        plt.savefig(config.CONFUSION_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        _plot_importance(pipe, X_te, y_te)

    print(f"Test F1 {metrics['test_f1']:.3f} | ROC-AUC {metrics['test_roc_auc']:.3f} "
          f"| missed real alarms: {metrics['confusion_matrix']['fn']}")
    return metrics


def _plot_importance(pipe, X_te, y_te):
    from sklearn.inspection import permutation_importance
    # Average precision is threshold-free, so it also credits features that only
    # shift probabilities without flipping a 0.5 decision. Correlated readings share
    # credit, so a low bar means "redundant here", not "useless".
    r = permutation_importance(pipe, X_te, y_te, scoring="average_precision", n_repeats=10,
                               random_state=config.RANDOM_STATE)
    order = np.argsort(r.importances_mean)[-12:]
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(config.ALL_FEATURES)[order], r.importances_mean[order])
    plt.xlabel("Permutation importance (drop in average precision)")
    plt.title("Which readings drive the decision")
    plt.savefig(config.IMPORTANCE_PATH, dpi=120, bbox_inches="tight")
    plt.close()
