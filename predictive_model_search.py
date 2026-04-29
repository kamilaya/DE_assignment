import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "ea_country_panel_clean.csv"
OUT_PATH = "predictive_model_scores.csv"


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


def main() -> None:
    df = pd.read_csv(DATA_PATH).sort_values(["country", "year"]).reset_index(drop=True)

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

    reg_models = {
        "linear": Pipeline([("pre", pre), ("model", LinearRegression())]),
        "ridge": Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-4, 4, 50)))]),
        "elastic": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                        alphas=np.logspace(-4, 2, 40),
                        cv=5,
                        max_iter=20000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "pca_ridge": Pipeline(
            [
                ("pre", pre),
                ("pca", PCA(n_components=0.95)),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 50))),
            ]
        ),
        "pls2": Pipeline([("pre", pre), ("model", PLSRegression(n_components=2))]),
        "pls3": Pipeline([("pre", pre), ("model", PLSRegression(n_components=3))]),
        "rf": Pipeline(
            [
                ("pre", pre),
                ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42)),
            ]
        ),
    }

    clf_models = {
        "logit_l2": Pipeline(
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
        ),
        "pca_logit": Pipeline(
            [
                ("pre", pre),
                ("pca", PCA(n_components=0.95)),
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
        ),
    }

    reg_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    clf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []

    for target in ["birth_rate", "mean_age_first_marriage"]:
        y = df[target]
        for model_name, model in reg_models.items():
            score = cross_val_score(model, X, y, cv=reg_cv, scoring="r2").mean()
            rows.append(
                {
                    "task": "regression",
                    "target": target,
                    "derived_target": "",
                    "model": model_name,
                    "metric": "r2_cv_mean",
                    "score": float(score),
                }
            )

    for target in ["birth_rate", "mean_age_first_marriage"]:
        derived_target = f"{target}_high"
        y_bin = (df[target] > df[target].median()).astype(int)
        for model_name, model in clf_models.items():
            score = cross_val_score(model, X, y_bin, cv=clf_cv, scoring="roc_auc").mean()
            rows.append(
                {
                    "task": "classification",
                    "target": target,
                    "derived_target": derived_target,
                    "model": model_name,
                    "metric": "auc_cv_mean",
                    "score": float(score),
                }
            )

    out = pd.DataFrame(rows).sort_values(["task", "target", "score"], ascending=[True, True, False])
    out.to_csv(OUT_PATH, index=False)

    print(out.round({"score": 4}).to_string(index=False))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
