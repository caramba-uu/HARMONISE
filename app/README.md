# HARMONISE app

A local web app (Gradio) that runs the HARMONISE models on one patient and shows, for every visit, the conformal
prediction (RRMS / SPMS) next to the Lorscheider criteria. Everything in the app is also available through an API.

> Research software. Not a medical device; do not use it for clinical decisions.

## Start

From the repository root, after `pixi install` and `pixi run merge`:

```bash
pixi run app
```

Open http://127.0.0.1:7860. Nothing you enter is stored. The app runs on your computer, and uploaded files are removed from Gradio's temporary cache after 10 minutes.

## Using the app

**1. Country and model.** Choosing the patient's country only suggests a model: the one the paper found best calibrated for that country, together with the confidence level used in the paper. You can pick any other model; a one-line description shows which data it was trained and calibrated on.

| Model | Trained on | Calibrated on |
|---|---|---|
| Hybrid XX | the other MSBase training countries | country XX |
| Country-specific XX | country XX | country XX |
| Global model | all MSBase training countries | all MSBase training countries |
| Swedish model | the Swedish MS Registry | the Swedish MS Registry |

**2. Patient**, in one of three ways:

- **Generate a dummy patient**: a synthetic *non-progressing*, *slowly progressing* or *progressing* patient with 3-20 visits. These are random numbers, not registry data.
- **Upload CSV**: see [File format](#file-format). Files with several patients get a patient picker. Example files are under "Example files" in the app and in [`examples/`](examples/).
- **Enter manually**: birth date, sex and diagnosis date. Then set the number of EDSS visits and relapses and press *Update form*. Each visit needs a date, an EDSS (0-10 in steps of 0.5) and a treatment; the pyramidal functional score is optional. Dates can be typed (YYYY-MM-DD) or picked from the calendar.

**3. Results**

- **Confidence level** (0-100 %): starts at the paper's level for the country. A label is predicted for a visit when its p-value is above the significance level (1 − confidence).
- **Disease course**: per visit, the model's prediction (RRMS ●, SPMS ✕, RRMS or SPMS ■, no prediction |), the Lorscheider criteria, the clinician's diagnosis (only if the uploaded file has `disease_course`) and EDSS.
- **p-values**: the RRMS and SPMS p-values over time, with the significance level.
- **p-value plane**: each visit as a point (RRMS p-value vs SPMS p-value). The shaded areas show which prediction a point gets.
- **Table**: everything per visit.
- **SHAP**: how each feature pushes the random forest towards SPMS, per visit. It is computed only from this patient's visits and the trees in the model.
- **Patient data**: the data the prediction was made on.
- **API**: a ready-to-run Python script and curl command for the patient, model and confidence level on screen, plus the response `/predict` returns for it. Use the copy buttons.

A Light / Dark switch is at the top right.

### How the inputs are used

- **Model features** (per visit, same as `1_data_cleaning.ipynb`): age at visit, sex, EDSS, age at diagnosis, diagnosis year, and treatment as first-line DMT / second-line DMT / none / stem cell (HSCT). Treatment `other` counts as none. Each visit is predicted from its own values; nothing is carried over between visits.
- **Relapses and the pyramidal score** are used only by the **Lorscheider criteria**. The Lorscheider criteria ([Lorscheider et al., Brain 2016](https://doi.org/10.1093/brain/aww173)) define transition to SPMS as confirmed disability progression independent of relapse activity: an increase in EDSS of 1.0 point if EDSS ≤5.5, or 0.5 points if EDSS ≥6.0, with a minimum EDSS of 4.0 and a pyramidal functional system (FS) score of at least 2, confirmed over at least three months. The FS criterion is applied when it is recorded for a visit; when it is missing, the remaining criteria are applied without it. In the code (`lorscheider_progression_index` in `harmonise_utils.py`), each visit is compared with the preceding visit, EDSS scores within 30 days after a relapse onset are not used, and progression is confirmed at the first visit at least three months (90 days) later. At least three visits are needed. The app marks SPMS from the progression visit on.

## File format

**Visits CSV**, one row per visit (the patient's fields are repeated on every row):

| Column | Required | Example |
|---|---|---|
| `patient_code` | if the file holds several patients | `P001` |
| `birth_date` | yes | `1980-05-01` |
| `sex` | yes | `F` or `M` |
| `diagnosis_date` | yes | `2005-03-01` |
| `visit_date` | yes | `2006-01-10` |
| `edss_score` | yes | `2.5` |
| `treatment_class` | yes | `first_line`, `second_line`, `none`, `hsct` or `other` |
| `edss_pyramidal` | no | `2` |
| `country` | no | `AU` (sets the country and suggested model) |
| `disease_course` | no | `RR` or `SP` (the clinician's diagnosis, shown for comparison) |

**Relapses CSV** (optional): `patient_code` and `relapse_date`.

The files in `dummy_data/` use the same layout, so `dummy_data/msbase_visits.csv` can be uploaded directly.

## API

The app exposes five endpoints. They are listed with their parameters under **Use via API** at the bottom of the page. After a prediction, the **API** tab in the results shows the calls for that exact patient and what each endpoint returns.

| Endpoint | Input | Returns |
|---|---|---|
| `/models` | - | all models with a description |
| `/suggest_model` | `country` (ISO code) | suggested model, the paper's confidence level and whether the country was well calibrated |
| `/predict` | `patient` (JSON), `model`, `confidence` | per-visit p-values, prediction sets and the Lorscheider criteria |
| `/predict_csv` | `visits_csv` (CSV text), `relapses_csv`, `patient_code`, `model`, `confidence` | same as `/predict` |
| `/explain` | `patient` (JSON), `model` | SHAP values per visit and feature (SPMS class) |

`model = ""` uses the suggested model for `patient["country"]` (the Global model if no country is given). `confidence = -1` uses the paper's level for that country (90 % if no country is given).

**Patient JSON**

```json
{
  "birth_date": "1980-05-01", "sex": "F", "diagnosis_date": "2005-03-01", "country": "AU",
  "visits": [
    {"visit_date": "2006-01-10", "edss_score": 2.0, "treatment_class": "first_line"},
    {"visit_date": "2007-02-12", "edss_score": 2.5, "treatment_class": "first_line", "edss_pyramidal": 2},
    {"visit_date": "2008-03-03", "edss_score": 4.5, "treatment_class": "second_line"}
  ],
  "relapses": ["2006-06-01"]
}
```

**Python** (`pip install gradio_client`):

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(patient, "", -1, api_name="/predict")
print(result["model_spms_from_visit"], result["lorscheider_spms_from_visit"])
for visit in result["visits"]:
    print(visit["date"], visit["p_rrms"], visit["p_spms"], visit["prediction_set"], visit["lorscheider"])

shap = client.predict(patient, "", api_name="/explain")
csv_result = client.predict(open("app/examples/progressing_visits.csv").read(),
                            open("app/examples/progressing_relapses.csv").read(), "", "", -1, api_name="/predict_csv")
```

**curl**: a call is two requests, one to start it and one to fetch the result.

```bash
EVENT_ID=$(curl -s -X POST http://127.0.0.1:7860/gradio_api/call/suggest_model \
  -H "Content-Type: application/json" -d '{"data": ["AR"]}' | python3 -c "import sys, json; print(json.load(sys.stdin)['event_id'])")
curl -s -N http://127.0.0.1:7860/gradio_api/call/suggest_model/$EVENT_ID
```

**Outputs** for the patient above (numbers shortened; lists cut after the first entry)

`/models`: all 36 models

```json
[{"model": "Hybrid AU", "description": "Trained on other countries, calibrated on Australia."}, ...]
```

`/suggest_model` with `"AR"`

```json
{"country": "AR", "model": "Hybrid CZ", "confidence": 93, "well_calibrated_in_paper": true}
```

`/predict` (and `/predict_csv`, which returns the same)

```json
{
  "model": "Hybrid AU", "confidence": 90, "significance": 0.1,
  "model_spms_from_visit": null, "lorscheider_spms_from_visit": null, "lorscheider_applicable": true,
  "visits": [{"visit": 1, "date": "2006-01-10", "edss": 2.0, "p_rrms": 0.796, "p_spms": 0.0007,
              "prediction_set": ["RRMS"], "lorscheider": "RRMS"}, ...]
}
```

`prediction_set` is `["RRMS"]`, `["SPMS"]`, `["RRMS", "SPMS"]` (undecided) or `[]` (no prediction). Visit numbers count from 1 in date order. `null` in `*_spms_from_visit` means no SPMS visit.

`/explain`: SHAP values for the SPMS class; `base_value` plus the SHAP values of a visit add up to its `rf_spms_score`

```json
{
  "model": "Hybrid AU", "class": "SPMS", "base_value": 0.5,
  "visits": [{"visit": 1, "rf_spms_score": 0.0,
              "shap": {"Age at visit": -0.087, "Sex": 0.007, "EDSS score": -0.307, "Age at diagnosis": -0.023,
                       "Diagnosis year": -0.035, "First-line DMT": -0.022, "Second-line DMT": 0.0006,
                       "No treatment": -0.033, "Stem cell treatment": 0.0}}, ...]
}
```

## Files

- `app.py`: layout, events and API endpoints
- `app_utils.py`: input parsing, feature engineering, prediction, Lorscheider criteria, SHAP and plots
- `dummy_patients.py`: synthetic patient generator. `python app/dummy_patients.py` regenerates `examples/`
- `examples/`: synthetic example CSVs (one per patient type)

The country → model suggestions and confidence levels come from `files/model_outputs/published_reference/country_summary.json`, which `3_analysis.ipynb` writes.
