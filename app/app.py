# HARMONISE app: per-visit RRMS/SPMS conformal predictions and the Lorscheider criteria for one patient.
#   pixi run app        (then open http://127.0.0.1:7860; the API is documented under "Use via API")
import io
import json
import os
import random
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

import app_utils as au  # noqa: E402
from dummy_patients import PROFILES, TREATMENTS, make_patient  # noqa: E402
from harmonise_utils import describe_model  # noqa: E402

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
EXAMPLES = sorted(os.path.join(EXAMPLES_DIR, f) for f in os.listdir(EXAMPLES_DIR) if f.endswith(".csv"))
DEFAULT_COUNTRY = "SE"
MAX_VISITS, MAX_RELAPSES, START_VISITS = 25, 15, 3
PRIVATE = {"api_visibility": "private"}  # UI callbacks are not part of the public API

CSS = """
.gradio-container {max-width: 1150px !important; margin: 0 auto !important;}
.step h2 {margin-top: 0.8em; margin-bottom: 0;}
/* number boxes: hide the browser's tiny spinners, STEPPER_JS adds larger - / + buttons */
input[type=number] {-moz-appearance: textfield; appearance: textfield;}
input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button {-webkit-appearance: none; margin: 0;}
.step-btn {min-width: 2.4rem; font-size: 1.3rem; line-height: 1; cursor: pointer; color: var(--body-text-color);
           background: var(--background-fill-secondary); border: 1px solid var(--input-border-color);}
.step-btn:hover {background: var(--border-color-primary);}
label.block:has(> .step-btn) {display: grid; grid-template-columns: auto 1fr auto; column-gap: 0;}
label.block:has(> .step-btn) > :not(input):not(.step-btn) {grid-column: 1 / -1; justify-self: start;}
label.block > .step-btn:first-of-type {border-radius: var(--input-radius) 0 0 var(--input-radius); border-right: none;}
label.block > .step-btn:last-of-type {border-radius: 0 var(--input-radius) var(--input-radius) 0; border-left: none;}
label.block > input[type=number] {border-radius: 0; text-align: center;}
.tab-like-container:has(> .step-btn) {height: 2.4rem !important;}
.tab-like-container > .step-btn {border: none; border-radius: 0;}
.tab-like-container > input[type=number] {font-size: var(--text-md) !important;}
"""

# wraps every number box (gr.Number and the box next to each gr.Slider) in - / + buttons; rows shown later get
# them too. Gradio reads the new value from the "input" event and a slider's release from "blur".
STEPPER_JS = """
<script>
(() => {
  const SELECTOR = "label.block > input[type=number], .tab-like-container > input[type=number]";
  function button(input, text, dir) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "step-btn"; b.textContent = text;
    b.setAttribute("aria-label", dir > 0 ? "Increase" : "Decrease");
    b.addEventListener("click", (e) => {
      e.preventDefault();
      if (input.disabled) return;
      if (input.value === "") input.value = input.min || 0;
      else if (dir > 0) input.stepUp(); else input.stepDown();
      input.dispatchEvent(new Event("input", {bubbles: true}));
      input.dispatchEvent(new Event("blur"));
    });
    return b;
  }
  function add(root) {
    root.querySelectorAll(SELECTOR).forEach((input) => {
      if (input.dataset.stepper) return;
      input.dataset.stepper = "1";
      input.before(button(input, "\\u2212", -1));
      input.after(button(input, "+", 1));
    });
  }
  new MutationObserver(() => add(document)).observe(document.documentElement, {childList: true, subtree: true});
})();
</script>
"""

READING = """
**Prediction sets.** For each visit the model gives a p-value for RRMS and one for SPMS. A label is predicted when
its p-value is above the significance level (1 − confidence). A visit can be **RRMS** (●), **SPMS** (✕),
**RRMS or SPMS** (■, the model can't decide at this confidence) or **No prediction** (|, unlike both).
Higher confidence means fewer mistakes but more undecided visits.

**Lorscheider criteria** (Lorscheider et al., Brain 2016). Transition to SPMS is confirmed disability progression
independent of relapse activity: an increase in EDSS of 1.0 point if EDSS ≤ 5.5, or 0.5 points if EDSS ≥ 6.0, with a
minimum EDSS of 4.0 and a pyramidal functional system (FS) score of at least 2, confirmed over at least three months.
The FS criterion is applied when it is recorded for a visit; when it is missing, the remaining criteria are applied
without it. Here each visit is compared with the preceding one, EDSS scores within 30 days after a relapse are not
used, and progression is confirmed at the first visit at least 90 days later. Needs at least 3 visits; SPMS is shown
from the progression visit on. Relapses and the FS score are only used here; the models themselves don't use them.

**SHAP** explains the random forest score that the conformal p-values are built on.
"""

API_INTRO = """
The results on this page are also available from code. Below is a script for **this patient, model and confidence
level**, and what each endpoint returns for it. All endpoints are listed with their parameters under **Use via API**
at the bottom of the page; more examples are in `app/README.md`.

| Endpoint | Input | Returns |
|---|---|---|
| `/models` | - | all models with a description |
| `/suggest_model` | country code | suggested model and the paper's confidence level |
| `/predict` | patient (JSON), model, confidence | per-visit p-values, prediction sets and the Lorscheider criteria |
| `/predict_csv` | visits CSV text, relapses CSV text, patient code, model, confidence | same as `/predict` |
| `/explain` | patient (JSON), model | SHAP values per visit and feature |

`model = ""` uses the suggested model for the patient's `country`; `confidence = -1` uses the paper's level.
"""

# plots updated while their tab is hidden keep the wrong size; a window resize makes Plotly re-measure them
RESIZE_JS = "() => { [50, 300, 800].forEach(t => setTimeout(() => window.dispatchEvent(new Event('resize')), t)); }"


def fail(e):
    raise gr.Error(str(e))


# ---------------------------------------------------------------- API endpoints

def api_models() -> list:
    """All available models with a one-line description."""
    return [{"model": m, "description": describe_model(m)} for m in au.MODELS]


def api_suggest_model(country: str) -> dict:
    """Suggested model and the paper's confidence level for a country (ISO code, e.g. 'AU')."""
    info = au.SUMMARY.get(country, {})
    return {"country": country, "model": au.suggested_model(country), "confidence": au.paper_confidence(country),
            "well_calibrated_in_paper": info.get("ready")}


def api_predict(patient: dict, model: str = "", confidence: float = -1) -> dict:
    """Per-visit p-values, prediction sets and the Lorscheider criteria for one patient.

    patient: {"birth_date", "sex" (F/M), "diagnosis_date", "country" (optional),
              "visits": [{"visit_date", "edss_score", "treatment_class", "edss_pyramidal" (optional)}],
              "relapses": [dates]}. Dates as YYYY-MM-DD.
    model: a name from /models, or "" for the country's suggested model.
    confidence: 0-100, or -1 for the paper's level for the country.
    """
    try:
        model, confidence = au.resolve(model, confidence, patient.get("country"))
        return au.api_result(au.predict(model, patient), confidence)
    except (ValueError, KeyError, FileNotFoundError) as e:
        fail(e)


def api_predict_csv(visits_csv: str, relapses_csv: str = "", patient_code: str = "", model: str = "",
                    confidence: float = -1) -> dict:
    """Same as /predict, with the patient given as CSV text in the layout of app/examples/*_visits.csv
    (and optionally *_relapses.csv). patient_code picks a patient if the file holds several."""
    try:
        visits, relapses = au.read_upload(io.StringIO(visits_csv), io.StringIO(relapses_csv) if relapses_csv else None)
        data = au.from_upload(visits, relapses, patient_code or visits["patient_code"].iloc[0])
        model, confidence = au.resolve(model, confidence, data.get("country"))
        return au.api_result(au.predict(model, data), confidence)
    except (ValueError, KeyError, IndexError, FileNotFoundError) as e:
        fail(e)


def api_explain(patient: dict, model: str = "") -> dict:
    """SHAP values (SPMS class) of the model's random forest for every visit of the patient."""
    try:
        model, _ = au.resolve(model, -1, patient.get("country"))
        return au.api_shap(au.predict(model, patient))
    except (ValueError, KeyError, FileNotFoundError) as e:
        fail(e)


# ---------------------------------------------------------------- UI callbacks

def on_country(country):
    model = au.suggested_model(country)
    return model, au.paper_confidence(country), au.suggestion_text(country)


def run(model, confidence, theme, data, progress=gr.Progress()):
    if data is None:
        return [gr.skip()] * 14
    try:
        progress(0.1, desc="Loading the model")
        au.get_model(model)
        progress(0.5, desc="Predicting")
        res = au.predict(model, data)
    except (ValueError, FileNotFoundError) as e:
        fail(e)
    progress(0.8, desc="Drawing the plots")
    info, table = au.patient_table(data)
    n = len(res["patient"])
    return (res, gr.update(visible=True), *redraw(res, confidence, theme), info, table,
            None, None, None, gr.update(maximum=max(n, 2), value=1, interactive=n > 1), gr.update(visible=False))


def redraw(res, confidence, theme):
    if res is None:
        return [gr.skip()] * 5
    return (au.summary_text(res, confidence), au.course_plot(res, confidence, theme),
            au.trajectory_plot(res, confidence, theme), au.plane_plot(res, confidence, theme),
            au.results_table(res, confidence))


def explain_ui(res, visit, theme, progress=gr.Progress()):
    if res is None:
        fail("Run a prediction first")
    progress(0.1, desc="Computing SHAP values")
    sv = au.shap_values(res)
    progress(0.8, desc="Drawing the plots")
    return sv, au.shap_heatmap(res, sv, theme), au.shap_waterfall(res, sv, min(int(visit), len(sv["values"])), theme)


def api_tab(res, data, model, confidence, country, request: gr.Request):
    # code snippets and the real response of each endpoint for the patient on screen
    if res is None or data is None:
        return [gr.skip()] * 6
    host = request.headers.get("host", "127.0.0.1:7860") if request else "127.0.0.1:7860"
    python, curl = au.api_snippets(res, data, model, confidence, country, f"http://{host}/")
    responses = [api_models(), api_suggest_model(country if country in au.SUMMARY else "AU"),
                 au.api_result(res, confidence)]
    return python, curl, *[json.dumps(r, indent=2) for r in responses]


def api_tab_explain(res, progress=gr.Progress()):
    # SHAP doesn't depend on the confidence level, so it is only refreshed for a new prediction
    if res is None:
        return gr.skip()
    progress(0.1, desc="Computing SHAP values")
    return json.dumps(au.api_shap(res), indent=2)


def redraw_shap(res, sv, visit, theme):
    if res is None or sv is None:
        return gr.skip(), gr.skip()
    return au.shap_heatmap(res, sv, theme), au.shap_waterfall(res, sv, visit, theme)


def generate(profile, n_visits):
    seed = random.randrange(10**6)
    return au.from_dummy(make_patient(profile, n_visits, seed)), f"Synthetic *{profile.lower()}* patient (seed {seed})."


def on_upload(visits_file, relapses_file):
    if visits_file is None:
        return None, gr.update(choices=[], value=None, visible=False)
    try:
        visits, relapses = au.read_upload(visits_file, relapses_file)
    except (ValueError, pd.errors.ParserError) as e:
        fail(e)
    patients = list(dict.fromkeys(visits["patient_code"]))
    return ({"visits": visits.to_dict("records"), "relapses": relapses.to_dict("records")},
            gr.update(choices=patients, value=None, visible=True, label=f"Patient ({len(patients)} in the file)"))


def on_patient(upload, code, country):
    if not upload or not code:
        return gr.skip(), gr.skip()
    data = au.from_upload(pd.DataFrame(upload["visits"]), pd.DataFrame(upload["relapses"]), code)
    return data, data["country"] if data.get("country") in au.SUMMARY else country


def show_rows(n_visits, n_relapses):
    nv = max(1, min(int(n_visits or 1), MAX_VISITS))
    nr = max(0, min(int(n_relapses or 0), MAX_RELAPSES))
    return [gr.update(visible=i < nv) for i in range(MAX_VISITS)] + [gr.update(visible=i < nr) for i in range(MAX_RELAPSES)]


def collect_manual(birth, sex, diagnosis, n_visits, n_relapses, *values):
    nv, nr = int(n_visits or 1), int(n_relapses or 0)
    visit_values, relapse_values = values[:4 * MAX_VISITS], values[4 * MAX_VISITS:]
    visits = []
    for i in range(nv):
        date, edss, treatment, pyramidal = visit_values[4 * i:4 * i + 4]
        if edss is not None and abs(edss * 2 - round(edss * 2)) > 1e-9:
            fail(f"Visit {i + 1}: EDSS goes in steps of 0.5")
        visits.append({"visit_date": date, "edss_score": edss, "treatment_class": treatment, "edss_pyramidal": pyramidal})
    relapses = [r for r in relapse_values[:nr] if r]
    return {"birth_date": birth, "sex": sex, "diagnosis_date": diagnosis, "visits": visits, "relapses": relapses}


# ---------------------------------------------------------------- layout

with gr.Blocks(title="HARMONISE", analytics_enabled=False, delete_cache=(600, 600)) as demo:
    current = gr.State(None)     # the patient being predicted
    result = gr.State(None)
    shap_state = gr.State(None)
    upload_state = gr.State(None)

    with gr.Row():
        gr.Markdown("# HARMONISE\nRRMS / SPMS prediction at every visit, with conformal prediction and the "
                    "Lorscheider criteria. A research tool, not for clinical decisions.", scale=6)
        theme = gr.Radio(["Light", "Dark"], value="Light", show_label=False, container=False, scale=1, min_width=170)

    # step 1
    gr.Markdown("## 1. Country and model", elem_classes="step")
    with gr.Row():
        with gr.Column():
            country = gr.Dropdown(au.COUNTRY_CHOICES, value=DEFAULT_COUNTRY, label="Patient's country",
                                  info="Only used to suggest the best model for this country.")
            suggestion = gr.Markdown(au.suggestion_text(DEFAULT_COUNTRY))
        with gr.Column():
            model = gr.Dropdown(au.MODEL_CHOICES, value=au.suggested_model(DEFAULT_COUNTRY), label="Model")
            model_info = gr.Markdown(describe_model(au.suggested_model(DEFAULT_COUNTRY)))

    # step 2
    gr.Markdown("## 2. Patient", elem_classes="step")
    with gr.Tabs():
        with gr.Tab("Generate a dummy patient"):
            with gr.Row(equal_height=True):
                profile = gr.Radio(list(PROFILES), value="Progressing", label="Type of patient", scale=3)
                dummy_n = gr.Slider(3, 20, value=10, step=1, label="Number of visits", scale=2)
            dummy_button = gr.Button("Generate and predict", variant="primary")
            dummy_note = gr.Markdown()
        with gr.Tab("Upload CSV"):
            gr.Markdown("One row per visit, in the same layout as the example files. Relapses are optional.")
            with gr.Row():
                visits_file = gr.File(label="Visits CSV", file_types=[".csv"], type="filepath", height=120)
                relapses_file = gr.File(label="Relapses CSV (optional)", file_types=[".csv"], type="filepath", height=120)
            patient_pick = gr.Dropdown([], label="Patient", visible=False)
            with gr.Accordion("Example files", open=False):
                gr.File(value=EXAMPLES, interactive=False, file_count="multiple", show_label=False)
        with gr.Tab("Enter manually"):
            with gr.Row():
                m_birth = gr.DateTime(label="Birth date", include_time=False, type="string")
                m_sex = gr.Radio(["F", "M"], label="Sex")
                m_diagnosis = gr.DateTime(label="Diagnosis date", include_time=False, type="string")
            with gr.Row(equal_height=False):
                m_nv = gr.Number(START_VISITS, precision=0, minimum=1, maximum=MAX_VISITS, label="Number of EDSS visits")
                m_nr = gr.Number(0, precision=0, minimum=0, maximum=MAX_RELAPSES, label="Number of relapses")
                with gr.Column(min_width=160):
                    m_set = gr.Button("Update form", size="md")
            gr.Markdown("**Visits**")
            visit_rows, visit_inputs = [], []
            for i in range(MAX_VISITS):
                first = i == 0
                with gr.Row(visible=i < START_VISITS) as row:
                    visit_inputs += [
                        gr.DateTime(label="Visit date", show_label=first, include_time=False, type="string", scale=3),
                        gr.Number(None, label="EDSS", show_label=first, minimum=0, maximum=10, step=0.5, scale=2,
                                  placeholder="0-10, steps of 0.5"),
                        gr.Dropdown(TREATMENTS, value="first_line", label="Treatment", show_label=first, scale=2),
                        gr.Number(None, label="Pyramidal FS", show_label=first, minimum=0, maximum=6, precision=0,
                                  scale=2, placeholder="optional"),
                    ]
                visit_rows.append(row)
            gr.Markdown("**Relapses** (only used by the Lorscheider criteria)")
            relapse_rows, relapse_inputs = [], []
            with gr.Row():
                for i in range(MAX_RELAPSES):
                    with gr.Column(visible=False, min_width=200) as col:
                        relapse_inputs.append(gr.DateTime(label=f"Relapse {i + 1}", include_time=False, type="string"))
                    relapse_rows.append(col)
            manual_button = gr.Button("Predict", variant="primary")

    # step 3
    gr.Markdown("## 3. Results", elem_classes="step")
    no_results = gr.Markdown("Generate, upload or enter a patient to see the results.")
    with gr.Column(visible=False) as results:
        with gr.Row(equal_height=False):
            confidence = gr.Slider(0, 100, value=au.paper_confidence(DEFAULT_COUNTRY), step=1,
                                   label="Confidence level (%)", scale=5)
            with gr.Column(scale=1, min_width=170):
                reset = gr.Button("Use paper's level", size="md")
        summary = gr.Markdown()
        with gr.Tabs() as result_tabs:
            with gr.Tab("Disease course"):
                course = gr.Plot(show_label=False)
            with gr.Tab("p-values"):
                trajectory = gr.Plot(show_label=False)
            with gr.Tab("p-value plane"):
                plane = gr.Plot(show_label=False)
            with gr.Tab("Table"):
                table = gr.Dataframe(interactive=False, show_label=False)
            with gr.Tab("SHAP"):
                explain_button = gr.Button("Explain the predictions")
                shap_heat = gr.Plot(show_label=False)
                shap_visit = gr.Slider(1, 2, value=1, step=1, label="Visit to explain")
                shap_fall = gr.Plot(show_label=False)
            with gr.Tab("Patient data"):
                patient_info = gr.Markdown()
                patient_data = gr.Dataframe(interactive=False, show_label=False)
            with gr.Tab("API"):
                gr.Markdown(API_INTRO)
                api_python = gr.Code(language="python", label="Python", interactive=False, max_lines=22, show_line_numbers=False)
                api_curl = gr.Code(language="shell", label="curl", interactive=False, max_lines=10, wrap_lines=True, show_line_numbers=False)
                gr.Markdown("**Responses** for this patient")
                with gr.Tabs():
                    api_out = {}
                    for name in ("models", "suggest_model", "predict", "predict_csv", "explain"):
                        with gr.Tab(f"/{name}"):
                            if name == "predict_csv":
                                gr.Markdown("Same response as `/predict`, for the patient in the CSV files.")
                            else:
                                api_out[name] = gr.Code(language="json", show_label=False, interactive=False,
                                                        max_lines=22, show_line_numbers=False)
        with gr.Accordion("How to read the results", open=False):
            gr.Markdown(READING)

    # ---------------------------------------------------------------- events
    run_inputs = [model, confidence, theme, current]
    run_outputs = [result, results, summary, course, trajectory, plane, table, patient_info, patient_data,
                   shap_state, shap_heat, shap_fall, shap_visit, no_results]
    plots = [summary, course, trajectory, plane, table]

    country.change(on_country, country, [model, confidence, suggestion], **PRIVATE)
    model.change(lambda m: describe_model(m), model, model_info, **PRIVATE).then(run, run_inputs, run_outputs, **PRIVATE)
    confidence.change(redraw, [result, confidence, theme], plots, **PRIVATE)
    result_tabs.select(None, None, None, js=RESIZE_JS, **PRIVATE)
    # every new prediction (and a new confidence level) refreshes the API examples and resizes the plots
    api_outputs = [api_python, api_curl, api_out["models"], api_out["suggest_model"], api_out["predict"]]
    result.change(api_tab, [result, current, model, confidence, country], api_outputs, **PRIVATE).then(
        None, None, None, js=RESIZE_JS, **PRIVATE).then(api_tab_explain, result, api_out["explain"], **PRIVATE)
    confidence.release(api_tab, [result, current, model, confidence, country], api_outputs, **PRIVATE)
    reset.click(au.paper_confidence, country, confidence, **PRIVATE)

    dummy_button.click(generate, [profile, dummy_n], [current, dummy_note], **PRIVATE).then(
        run, run_inputs, run_outputs, **PRIVATE)
    for f in (visits_file, relapses_file):
        f.change(on_upload, [visits_file, relapses_file], [upload_state, patient_pick], **PRIVATE)
    patient_pick.input(on_patient, [upload_state, patient_pick, country], [current, country], **PRIVATE).then(
        run, run_inputs, run_outputs, **PRIVATE)
    m_set.click(show_rows, [m_nv, m_nr], visit_rows + relapse_rows, **PRIVATE)
    manual_button.click(collect_manual, [m_birth, m_sex, m_diagnosis, m_nv, m_nr] + visit_inputs + relapse_inputs,
                        current, **PRIVATE).then(run, run_inputs, run_outputs, **PRIVATE)

    explain_button.click(explain_ui, [result, shap_visit, theme], [shap_state, shap_heat, shap_fall], **PRIVATE).then(
        None, None, None, js=RESIZE_JS, **PRIVATE)
    shap_visit.release(lambda r, s, v, t: redraw_shap(r, s, v, t)[1], [result, shap_state, shap_visit, theme],
                       shap_fall, **PRIVATE)

    # theme: switch Gradio's colours in the browser, redraw the plots in the matching colours
    theme.change(None, theme, None, js="(t) => { document.body.classList.toggle('dark', t === 'Dark'); }", **PRIVATE)
    theme.change(redraw, [result, confidence, theme], plots, **PRIVATE).then(
        redraw_shap, [result, shap_state, shap_visit, theme], [shap_heat, shap_fall], **PRIVATE)
    demo.load(None, None, theme, js="() => document.body.classList.contains('dark') ? 'Dark' : 'Light'", **PRIVATE)

    gr.api(api_models, api_name="models")
    gr.api(api_suggest_model, api_name="suggest_model")
    gr.api(api_predict, api_name="predict")
    gr.api(api_predict_csv, api_name="predict_csv")
    gr.api(api_explain, api_name="explain")


if __name__ == "__main__":
    # load scikit-learn and the default model while the server starts, so the first prediction is instant
    threading.Thread(target=au.get_model, args=(au.suggested_model(DEFAULT_COUNTRY),), daemon=True).start()
    # Source Sans Pro, the font of MSP-tracker; it ships with Gradio, so nothing is fetched from Google Fonts
    font = [gr.themes.LocalFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"]
    demo.launch(theme=gr.themes.Soft(text_size=gr.themes.sizes.text_lg, font=font), css=CSS, head=STEPPER_JS,
                share=False)
