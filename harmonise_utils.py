# Helpers shared by the notebooks and the app (model files, JSD, model assignment, Lorscheider criteria).
import joblib
import numpy as np
import pandas as pd

FEATURES = ["age_at_visit", "sex_label", "edss_score", "age_at_diagnosis", "year_diagnosis",
            "first_line_DMT", "second_line_DMT", "no_treatment", "stem_cell_treatment"]
FEATURE_NAMES = ["Age at visit", "Sex", "EDSS score", "Age at diagnosis", "Diagnosis year",
                 "First-line DMT", "Second-line DMT", "No treatment", "Stem cell treatment"]

REFERENCE_COUNTRY = "SE"
REFERENCE_MODEL = "Country-specific SE"

# Japan: Hybrid CZ and Hybrid BE give almost the same JSD (~0.185, difference < 0.001), so which one is
# lowest depends on the random uniform sample used for the JSD (CZ wins in ~70% of samples).
# We keep CZ, which is what the paper and its SHAP plots used.
NEAR_TIE_SWITCH = {"JP": {"Hybrid BE": "Hybrid CZ"}}

COUNTRY_NAMES = {
    "AE": "United Arab Emirates", "AR": "Argentina", "AU": "Australia", "BE": "Belgium", "BR": "Brazil",
    "CA": "Canada", "CH": "Switzerland", "CO": "Colombia", "CU": "Cuba", "CZ": "Czech Republic",
    "DK": "Denmark", "EE": "Estonia", "EG": "Egypt", "ES": "Spain", "GB": "United Kingdom", "GR": "Greece",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IN": "India", "IR": "Iran", "IT": "Italy",
    "JP": "Japan", "KW": "Kuwait", "LB": "Lebanon", "MT": "Malta", "MX": "Mexico", "NL": "Netherlands",
    "NZ": "New Zealand", "OM": "Oman", "PT": "Portugal", "RO": "Romania", "SA": "Saudi Arabia",
    "TN": "Tunisia", "TR": "Turkey", "UA": "Ukraine", "US": "United States of America", "SE": "Sweden",
}

LORSCHEIDER_COLUMNS = ["rrms_gt", "spms_gt", "transition_gt", "rrms_lor", "spms_lor", "transition_lor",
                       "rrms", "spms", "transition"]


def label(cc):
    return f"{COUNTRY_NAMES.get(cc, cc)} ({cc})"


def code_of(country_label):
    return country_label.split("(")[-1].rstrip(")")


def lambda_function(x):
    # mondrian condition: calibrate per class label
    return x[1]


def model_names(trainable):
    return [f"Hybrid {cc}" for cc in trainable] + ["Global model", REFERENCE_MODEL]


def model_file(models_dir, model_name):
    if model_name.startswith("Hybrid"):
        return f"{models_dir}/hybrid_{model_name.split()[1]}.joblib"
    if model_name == "Global model":
        return f"{models_dir}/global.joblib"
    if model_name == REFERENCE_MODEL:
        return f"{models_dir}/swedish.joblib"
    return f"{models_dir}/country_specific_{model_name.split()[1]}.joblib"


def load_model(path):
    # the published models were pickled from a script whose condition function was __main__.lambda_function_main,
    # so that name has to exist in __main__ before they can be unpickled
    import __main__
    if not hasattr(__main__, "lambda_function_main"):
        __main__.lambda_function_main = lambda_function
    return joblib.load(path)


def describe_model(model_name):
    if model_name.startswith("Hybrid"):
        return f"Trained on other countries, calibrated on {COUNTRY_NAMES[model_name.split()[1]]}."
    if model_name == "Global model":
        return "Trained and calibrated on all MSBase training countries together."
    if model_name == REFERENCE_MODEL:
        return "Trained and calibrated on the Swedish MS Registry (SMSReg)."
    return f"Trained and calibrated on {COUNTRY_NAMES[model_name.split()[1]]} only."


def shap_tag(model_name):
    return "global" if model_name == "Global model" else model_name.split()[-1]


# ---------------------------------------------------------------- JSD and model assignment

UNIFORM = np.random.default_rng(0).uniform(0, 1, 100_000)


def jsd(a, b, n_bins=10):
    from scipy.stats import entropy  # imported here so the app (which never needs it) starts faster
    combined = np.concatenate((a, b))
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)
    p, _ = np.histogram(a, bins=bins, density=True)
    q, _ = np.histogram(b, bins=bins, density=True)
    p = p / p.sum() + 1e-10
    q = q / q.sum() + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))


def as_arrays(entry):
    return np.asarray(entry["p_values"]), np.asarray(entry["y_label"]).astype(int)


def weighted_jsd(entry):
    # JSD of the true-label p-values against uniform, per class. RRMS is weighted by the SPMS fraction
    # and vice versa so the smaller class still counts.
    p, y = as_arrays(entry)
    if not (y == 1).any() or not (y == 0).any():
        return np.nan
    rows = np.arange(len(y))
    rrms = jsd(p[rows[y == 0], 0], UNIFORM)
    spms = jsd(p[rows[y == 1], 1], UNIFORM)
    spms_frac = (y == 1).mean()
    return rrms * spms_frac + spms * (1 - spms_frac)


def jsd_matrix(entry_for, countries, trainable):
    return pd.DataFrame({m: {label(cc): weighted_jsd(entry_for(m, cc)) for cc in countries}
                         for m in model_names(trainable)})


def assign_model(cc, row, trainable, verbose=True):
    if cc in trainable:
        return f"Hybrid {cc}"
    if cc == REFERENCE_COUNTRY:
        return REFERENCE_MODEL
    lowest = row.idxmin()
    switched = NEAR_TIE_SWITCH.get(cc, {}).get(lowest)
    if switched and switched in row.index:
        if verbose:
            print(f"{label(cc)}: lowest is {lowest} ({row[lowest]:.4f}), using {switched} ({row[switched]:.4f}) "
                  "instead - near tie, see NEAR_TIE_SWITCH")
        return switched
    return lowest


def assign_models(matrix, trainable, verbose=True):
    return {code_of(c): assign_model(code_of(c), row, trainable, verbose) for c, row in matrix.iterrows()}


def best_significance(p):
    # significance (0, 0.01, ..., 0.99) giving the most single-label predictions
    best, best_count = 0.0, -1
    for eps in np.arange(0, 1, 0.01):
        count = int((np.sum(p >= eps, axis=1) == 1).sum())
        if count > best_count:
            best, best_count = eps, count
    return best


# ---------------------------------------------------------------- Lorscheider criteria

def lorscheider_progression_index(visits, relapse_dates):
    # EDSS increase of >= 1.0 (baseline <= 5.5) or >= 0.5 (baseline > 5.5) to at least 4, no relapse
    # within 30 days, confirmed at the first visit >= 90 days later. A recorded pyramidal FS < 2 rules out
    # that baseline; if FS is missing it is ignored.
    edss = visits["edss_score"].tolist()
    dates = visits["visit_date"].tolist()
    pyramidal = visits["edss_pyramidal"].tolist()
    if len(edss) < 3:
        return None
    for i, baseline in enumerate(edss):
        if pd.notna(pyramidal[i]) and pyramidal[i] < 2:
            continue
        if i + 2 >= len(edss):
            break
        progression, progression_date = edss[i + 1], dates[i + 1]
        if progression <= baseline:
            continue
        step = 1.0 if baseline <= 5.5 else 0.5
        if progression < 4 or progression - baseline < step:
            continue
        if any(abs((progression_date - r).days) <= 30 for r in relapse_dates):
            continue
        for j in range(i + 2, len(edss)):
            if (dates[j] - progression_date).days >= 90:
                if edss[j] - baseline >= step:
                    return i + 1
                break
    return None


def model_visit_labels(frame, icp):
    # 0 = RRMS, 1 = SPMS, 2 = both, 3 = empty
    np.random.seed(0)
    p = icp.predict(frame[FEATURES].values)
    included = p > best_significance(p)
    size = included.sum(axis=1)
    return np.where(size == 2, 2, np.where(size == 0, 3, included.argmax(axis=1)))


def benchmark_country(test, relapses, icp):
    # Patient-level comparison of the Lorscheider criteria and the model on one test set.
    # Only patients with >= 3 visits; a patient counts as SPMS if any visit is classified SPMS.
    # Like the original code, the significance level is picked per patient group.
    test = test.sort_values(["patient_code", "visit_date"])
    test = test[test.groupby("patient_code")["visit_date"].transform("size") >= 3]
    label_sets = test.groupby("patient_code")["y_label"].agg(lambda s: frozenset(s))
    group_of = label_sets.map({frozenset({0}): "rrms", frozenset({1}): "spms", frozenset({0, 1}): "transition"})
    relapses_of = relapses.groupby("patient_code")["relapse_date"].apply(list).to_dict()

    counts, lor_diff, pred_diff = {}, [], []
    for group in ["rrms", "spms", "transition"]:
        sub = test[test["patient_code"].map(group_of) == group].copy()
        counts[f"{group}_gt"] = int(sub["patient_code"].nunique())
        if sub.empty:
            counts[f"{group}_lor"] = counts[group] = 0
            continue
        sub["model_label"] = model_visit_labels(sub, icp)
        n_lor = n_model = 0
        for patient, visits in sub.groupby("patient_code", sort=False):
            lor_idx = lorscheider_progression_index(visits, relapses_of.get(patient, []))
            preds = visits["model_label"].tolist()
            model_idx = preds.index(1) if 1 in preds else None
            n_lor += lor_idx is not None
            n_model += model_idx is not None
            if group == "transition":
                clinician_idx = visits["y_label"].tolist().index(1)
                if lor_idx is not None:
                    lor_diff.append(lor_idx - clinician_idx)
                if model_idx is not None:
                    pred_diff.append(model_idx - clinician_idx)
        counts[f"{group}_lor"], counts[group] = n_lor, n_model
    return {"counts": counts, "lor_diff": lor_diff, "pred_diff": pred_diff}
