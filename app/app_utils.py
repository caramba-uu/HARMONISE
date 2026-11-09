# Parsing, prediction, Lorscheider criteria, SHAP and plots for the HARMONISE app.
# Nothing is written to disk; the patient only lives in the browser session.
import json
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harmonise_utils import (COUNTRY_NAMES, FEATURE_NAMES, FEATURES, REFERENCE_MODEL, label, load_model,  # noqa: E402
                             lorscheider_progression_index, model_file)
from dummy_patients import TREATMENTS, VISIT_COLUMNS  # noqa: E402

MODELS_DIR = os.path.join(ROOT, "files", "models")
MANIFEST = os.path.join(ROOT, "files", "chunks", "manifest.json")
SUMMARY_FILE = os.path.join(ROOT, "files", "model_outputs", "published_reference", "country_summary.json")
DEFAULT_CONFIDENCE = 90
OTHER_COUNTRY = "Other / unknown"

with open(SUMMARY_FILE) as f:
    SUMMARY = json.load(f)

# colours per theme; the prediction set is also shown by marker shape, never by colour alone
THEMES = {
    "Light": {"ink": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e0", "mid": "#f0efec",
              "RR": "#2a78d6", "SP": "#eb6834", "RR|SP": "#1baf7a", "none": "#9e9d98"},
    "Dark": {"ink": "#f0efec", "muted": "#c3c2b7", "grid": "#3a3936", "mid": "#2a2a28",
             "RR": "#3987e5", "SP": "#d95926", "RR|SP": "#199e70", "none": "#8a8984"},
}
STATES = {
    "RR": {"symbol": "circle", "name": "RRMS"},
    "SP": {"symbol": "x", "name": "SPMS"},
    "RR|SP": {"symbol": "square", "name": "RRMS or SPMS"},
    "none": {"symbol": "line-ns-open", "name": "No prediction"},
}
FONT = 15
FONT_FAMILY = "Source Sans Pro, ui-sans-serif, system-ui, sans-serif"  # same as the app
HEIGHT = 460


# ---------------------------------------------------------------- countries and models

def available_models():
    on_disk = set(os.listdir(MODELS_DIR)) if os.path.isdir(MODELS_DIR) else set()
    with open(MANIFEST) as f:
        on_disk |= {os.path.basename(k) for k in json.load(f)}
    on_disk = {f for f in on_disk if f.endswith(".joblib")}
    hybrid = sorted(f[7:9] for f in on_disk if f.startswith("hybrid_"))
    specific = sorted(f[17:19] for f in on_disk if f.startswith("country_specific_"))
    return ([f"Hybrid {cc}" for cc in hybrid] + ["Global model", REFERENCE_MODEL]
            + [f"Country-specific {cc}" for cc in specific])


MODELS = available_models()


def model_title(name):
    return "Swedish model" if name == REFERENCE_MODEL else name


MODEL_CHOICES = [(model_title(m), m) for m in MODELS]
COUNTRY_CHOICES = [(label(cc), cc) for cc in sorted(SUMMARY, key=label)] + [(OTHER_COUNTRY, OTHER_COUNTRY)]


def suggested_model(country):
    return SUMMARY[country]["model"] if country in SUMMARY else "Global model"


def paper_confidence(country):
    if country not in SUMMARY:
        return DEFAULT_CONFIDENCE
    return int(round(100 * (1 - SUMMARY[country]["significance"])))


def suggestion_text(country):
    if country not in SUMMARY:
        return "No country selected: the **Global model** is suggested."
    text = f"Suggested model for {COUNTRY_NAMES.get(country, country)}: **{model_title(SUMMARY[country]['model'])}**"
    if not SUMMARY[country]["ready"]:
        text += "  \nNo model was well calibrated for this country in the paper; interpret the confidence with care."
    return text


@lru_cache(maxsize=3)
def get_model(name):
    path = model_file(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{os.path.relpath(path, ROOT)} is missing. Run `pixi run merge` first.")
    return load_model(path)


# ---------------------------------------------------------------- input
# a patient is passed around as {"birth_date", "sex", "diagnosis_date", "visits": [dict], "relapses": [date str]}

def parse_date(value, what):
    text = str(value).strip()[:10] if value is not None else ""
    if text in ("", "nan", "None", "NaT"):
        raise ValueError(f"{what} is missing")
    for fmt in ("%Y-%m-%d", "%m/%Y", "%Y-%m", "%d/%m/%Y", "%Y"):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            pass
    raise ValueError(f"{what}: could not read '{text}' (use YYYY-MM-DD)")


def build_patient(data):
    # same feature engineering as add_features() in 1_data_cleaning.ipynb
    birth = parse_date(data["birth_date"], "Birth date")
    diagnosis = parse_date(data["diagnosis_date"], "Diagnosis date")
    if data["sex"] not in ("F", "M"):
        raise ValueError("Sex must be F or M")
    if not data["visits"]:
        raise ValueError("Add at least one visit")
    rows = []
    for i, r in enumerate(data["visits"], 1):
        date = parse_date(r.get("visit_date"), f"Visit {i}: date")
        edss = pd.to_numeric(r.get("edss_score"), errors="coerce")
        if pd.isna(edss) or not 0 <= edss <= 10:
            raise ValueError(f"Visit {i}: EDSS must be a number between 0 and 10")
        treatment = str(r.get("treatment_class", "")).strip()
        if treatment not in TREATMENTS:
            raise ValueError(f"Visit {i}: treatment must be one of {', '.join(TREATMENTS)}")
        course = str(r.get("disease_course", "") or "").strip()
        rows.append({"visit_date": date, "edss_score": float(edss),
                     "edss_pyramidal": pd.to_numeric(r.get("edss_pyramidal"), errors="coerce"),
                     "treatment_class": treatment, "disease_course": course if course in ("RR", "SP") else ""})
    p = pd.DataFrame(rows).sort_values("visit_date").reset_index(drop=True)
    if (p["visit_date"] < birth).any() or diagnosis < birth:
        raise ValueError("Visit and diagnosis dates must be after the birth date")

    p["age_at_visit"] = (p["visit_date"] - birth).dt.days / 365.25
    p["sex_label"] = 1 if data["sex"] == "F" else 0
    p["age_at_diagnosis"] = (diagnosis - birth).days / 365.25
    p["year_diagnosis"] = diagnosis.year
    treatment = p["treatment_class"].replace({"other": "none"})
    p["first_line_DMT"] = (treatment == "first_line").astype(int)
    p["second_line_DMT"] = (treatment == "second_line").astype(int)
    p["no_treatment"] = (treatment == "none").astype(int)
    p["stem_cell_treatment"] = (treatment == "hsct").astype(int)
    p["years_since_diagnosis"] = (p["visit_date"] - diagnosis).dt.days / 365.25
    relapses = [parse_date(r, f"Relapse {i}") for i, r in enumerate(data["relapses"], 1)]
    return p, relapses


def from_dummy(patient):
    return {"birth_date": patient["birth_date"], "sex": patient["sex"], "diagnosis_date": patient["diagnosis_date"],
            "visits": patient["visits"].to_dict("records"), "relapses": patient["relapses"]["relapse_date"].tolist()}


def read_upload(visits_file, relapses_file):
    visits = pd.read_csv(visits_file, dtype=str)
    missing = {"birth_date", "sex", "diagnosis_date", "visit_date", "edss_score", "treatment_class"} - set(visits.columns)
    if missing:
        raise ValueError(f"The visits file is missing columns: {', '.join(sorted(missing))}")
    if "patient_code" not in visits.columns:
        visits["patient_code"] = "patient"
    relapses = pd.read_csv(relapses_file, dtype=str) if relapses_file else pd.DataFrame(columns=["patient_code", "relapse_date"])
    if "relapse_date" not in relapses.columns:
        raise ValueError("The relapses file needs a relapse_date column")
    if "patient_code" not in relapses.columns:
        relapses["patient_code"] = visits["patient_code"].iloc[0]
    return visits, relapses


def from_upload(visits, relapses, patient_code):
    v = visits[visits["patient_code"] == patient_code]
    first = v.iloc[0]
    cols = [c for c in VISIT_COLUMNS + ["disease_course"] if c in v.columns]
    return {"birth_date": first["birth_date"], "sex": first["sex"], "diagnosis_date": first["diagnosis_date"],
            "visits": v[cols].fillna("").to_dict("records"),
            "relapses": relapses.loc[relapses["patient_code"] == patient_code, "relapse_date"].dropna().tolist(),
            "country": first.get("country")}


def patient_table(data):
    # what the prediction was made on, shown in the "Patient data" tab
    visits = pd.DataFrame(data["visits"]).rename(columns={"visit_date": "Visit date", "edss_score": "EDSS",
                                                          "edss_pyramidal": "Pyramidal FS", "treatment_class": "Treatment",
                                                          "disease_course": "Clinician's diagnosis"})
    info = (f"Born {data['birth_date']} · Sex {data['sex']} · Diagnosed {data['diagnosis_date']} · "
            f"Relapses: {', '.join(map(str, data['relapses'])) or 'none'}")
    return info, visits


# ---------------------------------------------------------------- prediction

def predict(model_name, data):
    patient, relapses = build_patient(data)
    icp = get_model(model_name)
    np.random.seed(0)  # p-values are smoothed with random noise, keep them reproducible
    p = icp.predict(patient[FEATURES].values.astype(float))
    lor = lorscheider_progression_index(patient, relapses) if len(patient) >= 3 else None
    return {"model": model_name, "p": p.tolist(), "lorscheider": lor, "enough_visits": len(patient) >= 3,
            "patient": patient}


def prediction_sets(p, confidence):
    eps = 1 - confidence / 100
    inc = np.asarray(p) > eps
    return ["RR|SP" if a and b else "RR" if a else "SP" if b else "none" for a, b in inc]


def first_index(sets, state="SP"):
    return sets.index(state) if state in sets else None


def lorscheider_states(res):
    n = len(res["patient"])
    if not res["enough_visits"]:
        return ["none"] * n
    idx = res["lorscheider"]
    return ["SP" if idx is not None and i >= idx else "RR" for i in range(n)]


def summary_text(res, confidence):
    sets = prediction_sets(res["p"], confidence)
    dates = res["patient"]["visit_date"]
    at = lambda i: f"SPMS from visit {i + 1} ({dates[i]:%Y-%m-%d})"
    model_i = first_index(sets)
    model = at(model_i) if model_i is not None else "no visit classified as SPMS"
    if not res["enough_visits"]:
        lor = "needs at least 3 visits"
    else:
        lor = at(res["lorscheider"]) if res["lorscheider"] is not None else "no confirmed progression"
    return (f"**{model_title(res['model'])}** at {confidence:.0f} % confidence: {model}  \n"
            f"**Lorscheider criteria:** {lor}")


def results_table(res, confidence):
    patient, p = res["patient"], np.asarray(res["p"])
    table = pd.DataFrame({
        "Visit": np.arange(1, len(patient) + 1),
        "Date": patient["visit_date"].dt.strftime("%Y-%m-%d"),
        "EDSS": patient["edss_score"],
        "p RRMS": p[:, 0].round(3),
        "p SPMS": p[:, 1].round(3),
        "Model": [STATES[s]["name"] for s in prediction_sets(p, confidence)],
        "Lorscheider": [STATES[s]["name"] if s != "none" else "-" for s in lorscheider_states(res)],
    })
    if (patient["disease_course"] != "").any():
        table["Clinician's diagnosis"] = patient["disease_course"].map({"RR": "RRMS", "SP": "SPMS"}).fillna("-")
    return table


# ---------------------------------------------------------------- plots

def _layout(fig, theme, height=HEIGHT):
    t = THEMES[theme]
    fig.update_layout(height=height, autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": t["muted"], "size": FONT, "family": FONT_FAMILY},
                      margin={"l": 130, "r": 30, "t": 60, "b": 60},
                      hovermode="closest", hoverlabel={"font_size": FONT},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "x": 0, "xanchor": "left",
                              "font": {"color": t["ink"]}})
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False, linecolor=t["grid"], title_font={"color": t["ink"]})
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False, linecolor=t["grid"], title_font={"color": t["ink"]})
    return fig


def _x_range(x):
    return [min(0, float(x.min())) - 0.5, float(x.max()) + 0.5]


def _marker(states, theme, size=14):
    colors = [THEMES[theme][s] for s in states]
    return {"size": size, "symbol": [STATES[s]["symbol"] for s in states], "color": colors,
            "line": {"width": 2.5, "color": colors}}


def _state_legend(fig, theme, **where):
    for s in STATES:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=STATES[s]["name"],
                                 marker=_marker([s], theme, 12)), **where)


def _hover(patient, p, sets):
    return [f"Visit {i + 1} · {d:%Y-%m-%d}<br>EDSS {e}<br>p RRMS {a:.3f} · p SPMS {b:.3f}<br>{STATES[s]['name']}"
            for i, (d, e, (a, b), s) in enumerate(zip(patient["visit_date"], patient["edss_score"], p, sets))]


def _vlines(fig, res, sets, x, theme, **kw):
    # first SPMS visit of the Lorscheider criteria and of the model; the earlier label goes left, the later right
    t = THEMES[theme]
    lines = [(res["lorscheider"], "Lorscheider", t["ink"]), (first_index(sets), "Model", t["SP"])]
    lines = sorted((i, text, color) for i, text, color in lines if i is not None)
    same = len(lines) == 2 and lines[0][0] == lines[1][0]
    for k, (i, text, color) in enumerate(lines):
        fig.add_vline(x=x[i], line={"color": color, "dash": "dot", "width": 2}, **kw)
        if same and k == 1:
            continue
        side = "center" if same or len(lines) == 1 else ("right" if k == 0 else "left")
        fig.add_annotation(x=x[i], y=1.0, xref="x", yref="paper", yanchor="bottom", showarrow=False,
                           text=f"{'Model + Lorscheider' if same else text}: SPMS", font={"color": t["ink"], "size": 13},
                           xanchor=side, xshift={"center": 0, "right": -4, "left": 4}[side])


def course_plot(res, confidence, theme):
    t = THEMES[theme]
    patient = res["patient"]
    sets = prediction_sets(res["p"], confidence)
    x = patient["years_since_diagnosis"]
    tracks = [("Model", sets), ("Lorscheider", lorscheider_states(res))]
    if (patient["disease_course"] != "").any():
        tracks.append(("Clinician", [s if s else "none" for s in patient["disease_course"]]))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.1)
    for name, states in tracks:
        fig.add_trace(go.Scatter(
            x=x, y=[name] * len(x), mode="markers", showlegend=False, marker=_marker(states, theme, 16),
            text=[f"{name}: {STATES[s]['name']}<br>Visit {i + 1} · {d:%Y-%m-%d}"
                  for i, (s, d) in enumerate(zip(states, patient["visit_date"]))],
            hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    _state_legend(fig, theme, row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=patient["edss_score"], mode="lines+markers", showlegend=False,
                             line={"color": t["ink"], "width": 2}, marker={"size": 8, "color": t["ink"]},
                             hovertemplate="EDSS %{y}<extra></extra>"), row=2, col=1)
    _vlines(fig, res, sets, x, theme, row="all", col=1)
    names = [name for name, _ in tracks][::-1]
    fig.update_yaxes(categoryorder="array", categoryarray=names, range=[-0.5, len(names) - 0.5],
                     tickfont={"color": t["ink"]}, row=1, col=1)
    fig.update_yaxes(title="EDSS", range=[0, 10], dtick=2, row=2, col=1)
    fig.update_xaxes(range=_x_range(x))
    fig.update_xaxes(title="Years since diagnosis", row=2, col=1)
    return _layout(fig, theme, height=520)


def trajectory_plot(res, confidence, theme):
    t = THEMES[theme]
    patient, p = res["patient"], np.asarray(res["p"])
    sets = prediction_sets(p, confidence)
    x = patient["years_since_diagnosis"]
    hover = _hover(patient, p, sets)
    fig = go.Figure()
    for col, cls in [(0, "RR"), (1, "SP")]:
        marker = _marker(sets, theme, 12)
        marker["color"] = marker["line"]["color"] = t[cls]
        fig.add_trace(go.Scatter(x=x, y=p[:, col], mode="lines+markers", name=f"p-value {STATES[cls]['name']}",
                                 line={"color": t[cls], "width": 2}, marker=marker, text=hover,
                                 hovertemplate="%{text}<extra></extra>"))
    eps = 1 - confidence / 100
    fig.add_hline(y=eps, line={"dash": "dash", "color": t["muted"], "width": 1.5},
                  annotation_text=f"significance {eps:.2f}", annotation_position="top right",
                  annotation_font={"color": t["muted"], "size": 13})
    _vlines(fig, res, sets, x, theme)
    fig.update_yaxes(title="p-value", range=[-0.03, 1.03])
    fig.update_xaxes(title="Years since diagnosis", range=_x_range(x))
    return _layout(fig, theme)


def plane_plot(res, confidence, theme):
    t = THEMES[theme]
    patient, p = res["patient"], np.asarray(res["p"])
    sets = prediction_sets(p, confidence)
    eps = 1 - confidence / 100
    fig = go.Figure()
    for s, x0, x1, y0, y1 in [("RR", eps, 1.05, -0.05, eps), ("SP", -0.05, eps, eps, 1.05),
                              ("RR|SP", eps, 1.05, eps, 1.05), ("none", -0.05, eps, -0.05, eps)]:
        if x1 > x0 and y1 > y0:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=t[s], opacity=0.1,
                          line={"width": 0}, layer="below")
    fig.add_trace(go.Scatter(
        x=p[:, 0], y=p[:, 1], mode="lines+markers+text", showlegend=False, line={"color": t["grid"], "width": 1.5},
        text=[str(i + 1) for i in range(len(p))], textposition="top right", textfont={"color": t["ink"], "size": 13},
        marker=_marker(sets, theme, 13), hovertext=_hover(patient, p, sets), hovertemplate="%{hovertext}<extra></extra>"))
    _state_legend(fig, theme)
    fig.update_xaxes(title="p-value RRMS", range=[-0.05, 1.05])
    fig.update_yaxes(title="p-value SPMS", range=[-0.05, 1.05])
    return _layout(fig, theme, height=520)


# ---------------------------------------------------------------- SHAP

def shap_values(res):
    import shap
    rf = get_model(res["model"]).nc_function.model.model
    X = res["patient"][FEATURES].values.astype(float)
    # tree_path_dependent uses the node counts stored in the trees, so no background data is needed
    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
    values = np.asarray(explainer.shap_values(X))
    values = values[..., 1] if values.ndim == 3 else values[1]
    base = float(np.atleast_1d(explainer.expected_value)[1])
    return {"values": values.tolist(), "base": base, "prob": rf.predict_proba(X)[:, 1].tolist()}


def shap_heatmap(res, sv, theme):
    t = THEMES[theme]
    v = np.asarray(sv["values"])
    lim = float(np.abs(v).max()) or 1.0
    fig = go.Figure(go.Heatmap(
        z=v.T, x=[f"Visit {i + 1}" for i in range(len(v))], y=FEATURE_NAMES, zmid=0, zmin=-lim, zmax=lim,
        colorscale=[[0, t["RR"]], [0.5, t["mid"]], [1, t["SP"]]], xgap=2, ygap=2,
        colorbar={"title": {"text": "SHAP", "font": {"color": t["ink"]}}},
        hovertemplate="%{y} · %{x}<br>SHAP %{z:+.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont={"color": t["ink"]})
    fig = _layout(fig, theme)
    fig.update_layout(margin={"l": 170})
    return fig


def _feature_value(feature, value):
    if feature == "sex_label":
        return "F" if value == 1 else "M"
    if feature in ("first_line_DMT", "second_line_DMT", "no_treatment", "stem_cell_treatment"):
        return "yes" if value == 1 else "no"
    if feature == "year_diagnosis":
        return f"{value:.0f}"
    return f"{value:.1f}"


def shap_waterfall(res, sv, visit, theme):
    t = THEMES[theme]
    i = int(visit) - 1
    v = np.asarray(sv["values"])[i]
    x = res["patient"][FEATURES].iloc[i]
    order = np.argsort(np.abs(v))
    names = [f"{FEATURE_NAMES[j]} = {_feature_value(FEATURES[j], x.iloc[j])}" for j in order]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=["Average"] + names + [f"Visit {i + 1}"],
        x=[sv["base"]] + v[order].tolist() + [0], measure=["absolute"] + ["relative"] * len(v) + ["total"],
        increasing={"marker": {"color": t["SP"]}}, decreasing={"marker": {"color": t["RR"]}},
        totals={"marker": {"color": t["muted"]}}, connector={"line": {"color": t["grid"]}},
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_xaxes(title=f"Random forest SPMS score (average {sv['base']:.2f} → visit {i + 1}: {sv['prob'][i]:.2f})",
                     range=[0, 1])
    fig.update_yaxes(tickfont={"color": t["ink"]})
    fig = _layout(fig, theme, height=480)
    fig.update_layout(margin={"l": 230})
    return fig


# ---------------------------------------------------------------- API (plain JSON in and out)

def resolve(model, confidence, country=None):
    # empty model -> suggested model for the country; negative confidence -> the paper's level for the country
    model = model or suggested_model(country)
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose one of: {', '.join(MODELS)}")
    confidence = paper_confidence(country) if confidence is None or confidence < 0 else float(confidence)
    if not 0 <= confidence <= 100:
        raise ValueError("confidence must be between 0 and 100")
    return model, confidence


def api_result(res, confidence):
    patient, p = res["patient"], np.asarray(res["p"])
    sets = prediction_sets(p, confidence)
    lor = lorscheider_states(res)
    model_i = first_index(sets)
    return {
        "model": res["model"],
        "confidence": confidence,
        "significance": round(1 - confidence / 100, 4),
        "model_spms_from_visit": None if model_i is None else model_i + 1,
        "lorscheider_spms_from_visit": None if res["lorscheider"] is None else res["lorscheider"] + 1,
        "lorscheider_applicable": res["enough_visits"],
        "visits": [{"visit": i + 1, "date": f"{patient['visit_date'][i]:%Y-%m-%d}",
                    "edss": float(patient["edss_score"][i]), "p_rrms": float(p[i, 0]), "p_spms": float(p[i, 1]),
                    "prediction_set": {"RR": ["RRMS"], "SP": ["SPMS"], "RR|SP": ["RRMS", "SPMS"], "none": []}[sets[i]],
                    "lorscheider": STATES[lor[i]]["name"] if res["enough_visits"] else None}
                   for i in range(len(patient))],
    }


def api_shap(res):
    sv = shap_values(res)
    return {"model": res["model"], "class": "SPMS", "base_value": sv["base"],
            "visits": [{"visit": i + 1, "rf_spms_score": sv["prob"][i],
                        "shap": dict(zip(FEATURE_NAMES, map(float, sv["values"][i])))} for i in range(len(sv["values"]))]}


def api_patient(res, data, country=None):
    # the patient the app just used, as the JSON the API expects (normalised dates and numbers)
    visits = []
    for _, r in res["patient"].iterrows():
        v = {"visit_date": f"{r['visit_date']:%Y-%m-%d}", "edss_score": float(r["edss_score"]),
             "treatment_class": r["treatment_class"]}
        if pd.notna(r["edss_pyramidal"]):
            v["edss_pyramidal"] = int(r["edss_pyramidal"])
        if r["disease_course"]:
            v["disease_course"] = r["disease_course"]
        visits.append(v)
    patient = {"birth_date": f"{parse_date(data['birth_date'], 'Birth date'):%Y-%m-%d}", "sex": data["sex"],
               "diagnosis_date": f"{parse_date(data['diagnosis_date'], 'Diagnosis date'):%Y-%m-%d}"}
    if country in SUMMARY:
        patient["country"] = country
    patient["visits"] = visits
    patient["relapses"] = [f"{parse_date(r, 'Relapse'):%Y-%m-%d}" for r in data["relapses"]]
    return patient


def api_snippets(res, data, model, confidence, country, url):
    patient = api_patient(res, data, country)
    pretty = json.dumps(patient, indent=4)
    python = f'''from gradio_client import Client   # pip install gradio_client

client = Client("{url}")

patient = {pretty}

# all models, and the suggested model + paper's confidence level for a country
models = client.predict(api_name="/models")
suggestion = client.predict("{country if country in SUMMARY else 'AU'}", api_name="/suggest_model")

# per-visit predictions and the Lorscheider criteria (model "" = suggested model, confidence -1 = paper's level)
result = client.predict(patient, "{model}", {confidence:g}, api_name="/predict")

# SHAP values per visit and feature
shap = client.predict(patient, "{model}", api_name="/explain")

# the same from CSV files (layout of app/examples/*.csv)
# result = client.predict(open("visits.csv").read(), open("relapses.csv").read(), "", "{model}", {confidence:g}, api_name="/predict_csv")
'''
    body = json.dumps({"data": [patient, model, confidence]})
    curl = f'''URL={url.rstrip("/")}

# 1. start the call, 2. fetch the result (same pattern for every endpoint)
EVENT_ID=$(curl -s -X POST $URL/gradio_api/call/predict -H "Content-Type: application/json" \\
  -d '{body}' \\
  | python3 -c "import sys, json; print(json.load(sys.stdin)['event_id'])")
curl -s -N $URL/gradio_api/call/predict/$EVENT_ID
'''
    return python, curl
