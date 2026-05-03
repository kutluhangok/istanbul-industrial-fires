from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


FIGURES = Path("figures")
REPORTS = Path("reports")
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def _save(path: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / path, dpi=180, bbox_inches="tight")
    plt.close()


def build_figures(clean: pd.DataFrame, istanbul: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    clean = clean.copy()
    clean["year"] = clean["year"].astype(int)
    clean["tarih_parsed"] = pd.to_datetime(clean["tarih_parsed"])
    istanbul["date"] = pd.to_datetime(istanbul["date"])

    monthly = clean.groupby(["year", "month"]).size().reset_index(name="count")
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="month", y="count", hue="year", marker="o", palette="tab10")
    plt.xticks(range(1, 13))
    plt.title("Monthly Industrial Fire and Explosion Incidents")
    plt.xlabel("Month")
    plt.ylabel("Incident count")
    _save("01_monthly_fire_counts.png")

    top_cities = clean["il"].replace("", np.nan).dropna().value_counts().head(15).sort_values()
    plt.figure(figsize=(9, 6))
    top_cities.plot(kind="barh", color="#2f6f73")
    plt.title("Top 15 Cities by Incident Count")
    plt.xlabel("Incident count")
    _save("02_top_cities.png")

    sector_counts = clean["sektor_std"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sector_counts.sort_values().plot(kind="barh", ax=axes[0], color="#6c8ebf")
    axes[0].set_title("Sector Counts")
    axes[0].set_xlabel("Incident count")
    sector_counts.head(8).plot(kind="pie", ax=axes[1], autopct="%1.0f%%", startangle=90)
    axes[1].set_ylabel("")
    axes[1].set_title("Top Sector Shares")
    _save("03_sector_distribution.png")

    yearly_type = clean.groupby(["year", "olay_turu"]).size().unstack(fill_value=0)
    yearly_type.plot(kind="bar", figsize=(10, 5), color=["#d95f02", "#1b9e77", "#7570b3"])
    plt.title("Fire vs Explosion by Year")
    plt.xlabel("Year")
    plt.ylabel("Incident count")
    _save("04_fire_vs_explosion_yearly.png")

    days = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    dow_counts = clean["day_of_week"].dropna().astype(int).value_counts().reindex(range(7), fill_value=0)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=days, y=dow_counts.values, color="#4e79a7")
    plt.xticks(rotation=25, ha="right")
    plt.title("Incidents by Day of Week")
    plt.ylabel("Incident count")
    _save("05_day_of_week.png")

    severity_counts = clean["severity"].value_counts().reindex(["low", "medium", "high"]).dropna()
    plt.figure(figsize=(7, 5))
    sns.barplot(x=severity_counts.index, y=severity_counts.values, palette=["#59a14f", "#f28e2b", "#e15759"])
    plt.title("Severity Distribution")
    plt.ylabel("Incident count")
    _save("06_severity_distribution.png")

    city_year_path = Path("data/processed/city_year_osb_panel.xlsx")
    if city_year_path.exists():
        panel = pd.read_excel(city_year_path)
    else:
        panel = clean.groupby(["il", "year"]).size().reset_index(name="incident_count")

    latest_rates = (
        panel.dropna(subset=["incidents_per_1000_parcels"])
        .groupby("il")
        .agg(incident_count=("incident_count", "sum"), osb_parcels=("osb_parcels", "first"))
        .query("osb_parcels >= 100")
    )
    latest_rates["incidents_per_1000_parcels"] = latest_rates["incident_count"] / latest_rates["osb_parcels"] * 1000
    top_rates = latest_rates.sort_values("incidents_per_1000_parcels", ascending=False).head(15).sort_values("incidents_per_1000_parcels")
    plt.figure(figsize=(9, 6))
    plt.barh(top_rates.index, top_rates["incidents_per_1000_parcels"], color="#e15759")
    plt.title("Exposure-Adjusted Incidents by Province")
    plt.xlabel("Incidents per 1,000 OSB parcels, 2017-2023")
    _save("07_istanbul_districts.png")

    osb_comp = pd.crosstab(clean["has_city_osb_exposure"], clean["severity"], normalize="index")
    osb_comp = osb_comp.reindex(columns=["low", "medium", "high"], fill_value=0)
    osb_comp.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#59a14f", "#f28e2b", "#e15759"])
    plt.title("Severity Mix: Provinces With vs Without OSB Exposure")
    plt.xlabel("Province has OSB exposure")
    plt.ylabel("Share")
    _save("08_osb_vs_non_osb_severity.png")

    province_totals = clean.groupby("il").agg(
        incident_count=("Tarih", "count"),
        osb_parcels=("osb_parcels", "first"),
        osb_area_hectare=("osb_area_hectare", "first"),
    ).query("osb_parcels > 0")
    plt.figure(figsize=(8, 5))
    sns.regplot(data=province_totals, x="osb_parcels", y="incident_count", scatter_kws={"alpha": 0.65}, color="#4e79a7")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("OSB Parcels vs Province Incident Count")
    plt.xlabel("OSB parcels (log)")
    plt.ylabel("Incidents, 2017-2023 (log)")
    _save("09_temp_vs_fire_count.png")

    plt.figure(figsize=(8, 5))
    sns.regplot(data=province_totals, x="osb_area_hectare", y="incident_count", scatter_kws={"alpha": 0.65}, color="#8e6c8a")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("OSB Area vs Province Incident Count")
    plt.xlabel("OSB area, hectare (log)")
    plt.ylabel("Incidents, 2017-2023 (log)")
    _save("10_humidity_vs_fire_count.png")

    ignition = clean["Tutuşturma Kaynağı"].replace({"": np.nan, "-": np.nan, "Bilinmeyen": np.nan}).dropna().value_counts().head(15).sort_values()
    plt.figure(figsize=(9, 6))
    ignition.plot(kind="barh", color="#edc948")
    plt.title("Top Ignition Sources")
    plt.xlabel("Incident count")
    _save("11_ignition_sources.png")

    cause = clean["Oluş Biçimleri"].replace({"": np.nan, "-": np.nan}).dropna().value_counts().head(15).sort_values()
    plt.figure(figsize=(9, 6))
    cause.plot(kind="barh", color="#b07aa1")
    plt.title("Top Incident Formation Patterns")
    plt.xlabel("Incident count")
    _save("12_incident_causes.png")


def hypothesis_tests(clean: pd.DataFrame, istanbul: pd.DataFrame) -> dict:
    REPORTS.mkdir(exist_ok=True)
    clean = clean.copy()
    istanbul = istanbul.copy()
    clean["year"] = clean["year"].astype(int)
    clean["month"] = clean["month"].astype(int)
    istanbul["date"] = pd.to_datetime(istanbul["date"])

    monthly_counts = clean.groupby(["year", "month"]).size().reset_index(name="count")
    groups = [monthly_counts[monthly_counts["month"] == m]["count"].values for m in range(1, 13)]
    groups = [g for g in groups if len(g) > 0]
    h_stat, h_p = stats.kruskal(*groups)
    n = len(monthly_counts)
    k = len(groups)
    epsilon_sq = (h_stat - k + 1) / (n - k) if n > k else np.nan

    province_counts = clean.groupby("il").agg(
        fire_count=("Tarih", "count"),
        osb_parcels=("osb_parcels", "first"),
        osb_area_hectare=("osb_area_hectare", "first"),
        has_city_osb_exposure=("has_city_osb_exposure", "first"),
    ).reset_index()
    osb_counts = province_counts[province_counts["has_city_osb_exposure"]]["fire_count"]
    non_osb_counts = province_counts[~province_counts["has_city_osb_exposure"]]["fire_count"]
    if len(osb_counts) and len(non_osb_counts):
        u_stat, u_p = stats.mannwhitneyu(osb_counts, non_osb_counts, alternative="greater")
    else:
        u_stat, u_p = np.nan, np.nan

    exposure_subset = province_counts.query("osb_parcels > 0 and osb_area_hectare > 0").copy()
    exposure_corr = {
        "osb_parcels_spearman": stats.spearmanr(exposure_subset["fire_count"], exposure_subset["osb_parcels"]).statistic if len(exposure_subset) else np.nan,
        "osb_parcels_p_value": stats.spearmanr(exposure_subset["fire_count"], exposure_subset["osb_parcels"]).pvalue if len(exposure_subset) else np.nan,
        "osb_area_spearman": stats.spearmanr(exposure_subset["fire_count"], exposure_subset["osb_area_hectare"]).statistic if len(exposure_subset) else np.nan,
        "osb_area_p_value": stats.spearmanr(exposure_subset["fire_count"], exposure_subset["osb_area_hectare"]).pvalue if len(exposure_subset) else np.nan,
    }

    ct_sector = pd.crosstab(clean["sektor_std"], clean["severity"])
    chi2_sector, p_sector, dof_sector, _ = chi2_contingency(ct_sector)
    ct_osb = pd.crosstab(clean["has_city_osb_exposure"], clean["severity"])
    chi2_osb, p_osb, dof_osb, _ = chi2_contingency(ct_osb)

    results = {
        "H1_seasonality_kruskal": {"H": float(h_stat), "p_value": float(h_p), "epsilon_squared": float(epsilon_sq)},
        "H2_osb_exposure_mannwhitney": {
            "U": float(u_stat) if pd.notna(u_stat) else None,
            "p_value": float(u_p) if pd.notna(u_p) else None,
            "osb_province_median": float(osb_counts.median()) if len(osb_counts) else None,
            "non_osb_province_median": float(non_osb_counts.median()) if len(non_osb_counts) else None,
        },
        "H3_osb_exposure_correlations": {k: (float(v) if pd.notna(v) else None) for k, v in exposure_corr.items()},
        "H4_sector_severity_chi_square": {"chi2": float(chi2_sector), "p_value": float(p_sector), "dof": int(dof_sector)},
        "H4_province_osb_exposure_severity_chi_square": {"chi2": float(chi2_osb), "p_value": float(p_osb), "dof": int(dof_osb)},
    }
    (REPORTS / "hypothesis_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def run_ml(clean: pd.DataFrame, istanbul: pd.DataFrame) -> pd.DataFrame:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    clean = clean.copy()
    istanbul_weather = istanbul[
        [
            "source_file",
            "Tarih",
            "Firma İsmi",
            "ilce",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "windspeed_10m_max",
            "precipitation_sum",
            "extreme_heat",
            "low_humidity",
        ]
    ].copy()
    df_ml = clean.merge(
        istanbul_weather,
        on=["source_file", "Tarih", "Firma İsmi", "ilce"],
        how="left",
        suffixes=("", "_weather"),
    )
    for col in ["extreme_heat", "low_humidity"]:
        df_ml[col] = df_ml[col].fillna(False).astype(bool)

    boolean_features = ["is_weekend", "is_holiday", "is_istanbul", "has_osb", "extreme_heat", "low_humidity"]
    for col in boolean_features:
        df_ml[col] = df_ml[col].fillna(False).astype(int)

    features = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_holiday",
        "is_istanbul",
        "has_osb",
        "osb_count",
        "osb_area_hectare",
        "osb_parcels",
        "osb_operational_count",
        "osb_operational_area_hectare",
        "osb_operational_parcels",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "windspeed_10m_max",
        "precipitation_sum",
        "extreme_heat",
        "low_humidity",
        "sektor_std",
        "olay_turu",
    ]
    df_ml = df_ml.dropna(subset=["severity"]).copy()
    X = df_ml[features]
    le = LabelEncoder()
    y = le.fit_transform(df_ml["severity"])

    numeric_features = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "osb_count",
        "osb_area_hectare",
        "osb_parcels",
        "osb_operational_count",
        "osb_operational_area_hectare",
        "osb_operational_parcels",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "windspeed_10m_max",
        "precipitation_sum",
    ]
    categorical_features = ["sektor_std", "olay_turu"]
    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("bool", SimpleImputer(strategy="constant", fill_value=False), boolean_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(n_estimators=150, learning_rate=0.05, random_state=42, eval_metric="mlogloss")
    except Exception:
        pass

    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify, random_state=42)

    cv_splits = min(5, pd.Series(y).value_counts().min())
    cv = StratifiedKFold(n_splits=max(2, cv_splits), shuffle=True, random_state=42)
    rows = []
    best_name = None
    best_score = -np.inf
    fitted = {}
    for name, clf in models.items():
        model = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        report = classification_report(y_test, model.predict(X_test), target_names=le.classes_, output_dict=True, zero_division=0)
        rows.append(
            {
                "model": name,
                "cv_f1_macro_mean": scores.mean(),
                "cv_f1_macro_std": scores.std(),
                "test_f1_macro": report["macro avg"]["f1-score"],
                "test_accuracy": report["accuracy"],
            }
        )
        fitted[name] = model
        if scores.mean() > best_score:
            best_name = name
            best_score = scores.mean()

    comparison = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)
    comparison.to_excel(REPORTS / "model_comparison.xlsx", index=False)

    best_model = fitted[best_name]
    try:
        import shap

        transformed = best_model.named_steps["prep"].transform(X_test)
        prep = best_model.named_steps["prep"]
        cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(categorical_features))
        feature_names = numeric_features + cat_names + boolean_features
        clf = best_model.named_steps["clf"]
        if best_name in {"Random Forest", "XGBoost"}:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(transformed)
            shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
            plt.savefig(FIGURES / "14_shap_summary.png", bbox_inches="tight", dpi=180)
            plt.close()
    except Exception as exc:
        (REPORTS / "shap_warning.txt").write_text(str(exc), encoding="utf-8")

    return comparison


if __name__ == "__main__":
    clean_df = pd.read_excel("data/processed/kmo_incidents_clean.xlsx")
    istanbul_df = pd.read_excel("data/processed/istanbul_enriched.xlsx")
    build_figures(clean_df, istanbul_df)
    print(json.dumps(hypothesis_tests(clean_df, istanbul_df), indent=2, ensure_ascii=False))
    print(run_ml(clean_df, istanbul_df).to_string(index=False))
