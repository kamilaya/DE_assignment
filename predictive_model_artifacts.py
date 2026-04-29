import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "ea_country_panel_clean.csv"
SCORES_PATH = "predictive_model_scores.csv"
PREDICTIONS_PATH = "predictive_best_model_predictions.csv"


def build_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )


def save_score_chart(scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    reg = scores[scores["task"] == "regression"].copy()
    clf = scores[scores["task"] == "classification"].copy()

    for ax, subset, title, xlabel in [
        (axes[0], reg, "Regression Model Performance", "Cross-validated R^2"),
        (axes[1], clf, "Classification Model Performance", "Cross-validated AUC"),
    ]:
        labels = [f"{row.target}\n{row.model}" for row in subset.itertuples()]
        values = subset["score"].to_numpy()
        ypos = np.arange(len(labels))
        ax.barh(ypos, values, color="#4C78A8")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=8)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig("predictive_model_scores.png", dpi=160)
    plt.close(fig)


def save_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(y_true, y_pred, alpha=0.75, color="#4C78A8", edgecolors="none")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Cross-validated prediction")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def save_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, filename: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(fpr, tpr, color="#F58518", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH).sort_values(["country", "year"]).reset_index(drop=True)
    scores = pd.read_csv(SCORES_PATH)

    num_features = [
        "year",
        "hpi_index",
        "hicp_index",
        "gdp_growth",
        "unemp_rate",
        "hpi_growth",
        "hicp_growth",
        "hpi_growth_lag1",
        "hicp_growth_lag1",
        "gdp_growth_lag1",
        "unemp_rate_lag1",
    ]
    cat_features = ["country"]
    X = df[num_features + cat_features]

    pre = build_preprocessor(num_features, cat_features)

    birth_reg = Pipeline([("pre", pre), ("model", PLSRegression(n_components=3))])
    marriage_reg = Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-4, 4, 50)))])

    birth_clf = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegressionCV(
                    Cs=20,
                    cv=5,
                    max_iter=20000,
                    scoring="roc_auc",
                    l1_ratios=(0,),
                    use_legacy_attributes=False,
                ),
            ),
        ]
    )
    marriage_clf = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegressionCV(
                    Cs=20,
                    cv=5,
                    max_iter=20000,
                    scoring="roc_auc",
                    l1_ratios=(0,),
                    use_legacy_attributes=False,
                ),
            ),
        ]
    )

    reg_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    clf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_birth = df["birth_rate"].to_numpy()
    y_marriage = df["mean_age_first_marriage"].to_numpy()
    y_birth_high = (df["birth_rate"] > df["birth_rate"].median()).astype(int).to_numpy()
    y_marriage_high = (df["mean_age_first_marriage"] > df["mean_age_first_marriage"].median()).astype(int).to_numpy()

    birth_pred = cross_val_predict(birth_reg, X, y_birth, cv=reg_cv)
    marriage_pred = cross_val_predict(marriage_reg, X, y_marriage, cv=reg_cv)
    birth_prob = cross_val_predict(birth_clf, X, y_birth_high, cv=clf_cv, method="predict_proba")[:, 1]
    marriage_prob = cross_val_predict(marriage_clf, X, y_marriage_high, cv=clf_cv, method="predict_proba")[:, 1]

    preds = pd.DataFrame(
        {
            "country": df["country"],
            "year": df["year"],
            "birth_rate_actual": y_birth,
            "birth_rate_pred_cv": birth_pred,
            "birth_rate_high_actual": y_birth_high,
            "birth_rate_high_prob_cv": birth_prob,
            "mean_age_first_marriage_actual": y_marriage,
            "mean_age_first_marriage_pred_cv": marriage_pred,
            "mean_age_first_marriage_high_actual": y_marriage_high,
            "mean_age_first_marriage_high_prob_cv": marriage_prob,
        }
    )
    preds.to_csv(PREDICTIONS_PATH, index=False)

    save_score_chart(scores)
    save_regression_scatter(
        y_birth,
        birth_pred,
        "Birth Rate: Actual vs Predicted",
        "birth_rate_actual_vs_predicted.png",
    )
    save_regression_scatter(
        y_marriage,
        marriage_pred,
        "Marriage Age: Actual vs Predicted",
        "marriage_age_actual_vs_predicted.png",
    )
    save_roc(
        y_birth_high,
        birth_prob,
        "Birth Rate High: ROC Curve",
        "birth_rate_high_roc.png",
    )
    save_roc(
        y_marriage_high,
        marriage_prob,
        "Marriage Age High: ROC Curve",
        "marriage_age_high_roc.png",
    )

    print("Saved predictive artifacts:")
    print("- predictive_model_scores.png")
    print("- birth_rate_actual_vs_predicted.png")
    print("- marriage_age_actual_vs_predicted.png")
    print("- birth_rate_high_roc.png")
    print("- marriage_age_high_roc.png")
    print(f"- {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
