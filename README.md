# HARMONISE

Code, models and an interactive app for the paper *HARMONISE: an uncertainty-aware framework for harmonising disease-course predictions across international clinical registries*.

At every hospital visit, HARMONISE classifies a person with multiple sclerosis as relapsing-remitting (RRMS) or secondary progressive (SPMS). It uses Mondrian conformal predictors built on random forests, trained on the Swedish MS Registry (SMSReg) and MSBase. Conformal prediction gives each visit a p-value per class, so every prediction comes with a confidence level. The models are compared with the rule-based Lorscheider criteria for SPMS.

> Research software. Not a medical device; do not use it for clinical decisions.

## Quick start

You need [pixi](https://pixi.sh) (`curl -fsSL https://pixi.sh/install.sh | bash`) and git.

```bash
git clone https://github.com/caramba-uu/HARMONISE.git
cd HARMONISE
pixi install      # creates the environment (Python 3.12 + pinned packages)
pixi run merge    # rebuilds the large model and result files from files/chunks/ (about 2 GB, run once)
pixi run app      # starts the app at http://127.0.0.1:7860
pixi run lab      # or open the notebooks in JupyterLab
```

## The app

`pixi run app` opens a local web app for a single patient:

1. **Country and model.** The country is only used to suggest the best calibrated model for it (and the confidence level used in the paper). Any model can be selected.
2. **Patient.** Generate a synthetic dummy patient (non-, slowly or fast progressing), upload a CSV, or enter visits and relapses by hand.
3. **Results.** For each visit: the model's prediction next to the Lorscheider criteria, the p-values, a table and SHAP feature importance. A confidence slider (0-100 %) updates the predictions.

All results are also available through an API (`/predict`, `/predict_csv`, `/explain`, `/models`, `/suggest_model`). See [`app/README.md`](app/README.md) for the input format, how the inputs are used, and API examples.

## Models

All models are in `files/models/`. The labels match the paper.

| File | Label in paper / app | What it is |
|---|---|---|
| `swedish.joblib` | Country-specific SE (Swedish model) | Trained and calibrated on SMSReg |
| `global.joblib` | Global model | Trained and calibrated on all MSBase training countries pooled |
| `country_specific_XX.joblib` | Country-specific XX | Trained and calibrated only on country XX |
| `hybrid_XX.joblib` | Hybrid XX | Trained on the other MSBase training countries, calibrated on country XX |

`XX` is the ISO country code. The 17 MSBase training countries are AU, BE, CA, CH, CZ, EG, ES, GB, IR, IT, KW, LB, NL, PT, TN, TR and US. Each test country is assigned the model with the lowest divergence (JSD) between its p-values and the uniform distribution. The assignment, divergence, readiness and significance level per country are in `files/model_outputs/published_reference/country_summary.json`.

Each model is a `nonconformist` `IcpClassifier`. Load it with the helper, which supplies the condition function the published models were pickled with:

```python
from harmonise_utils import FEATURES, load_model
icp = load_model("files/models/hybrid_AU.joblib")
p_values = icp.predict(X)   # X: visits x FEATURES, columns [RRMS, SPMS]
```

The nine features, in order, are `age_at_visit, sex_label (F=1, M=0), edss_score, age_at_diagnosis, year_diagnosis, first_line_DMT, second_line_DMT, no_treatment, stem_cell_treatment`. Treatment class `other` counts as no treatment. See `add_features` in `1_data_cleaning.ipynb`.

## Reproducing the paper

Open `3_analysis.ipynb` and run it, with `MODEL_OUTPUTS_DIR = "files/model_outputs"` (the default). It reads the published results and writes the figures and tables to `output/manuscript/`. SHAP figures are skipped because the published SHAP values are not shared.

## Running on your own data

1. Put your data in `RAW_DIR` (set in `1_data_cleaning.ipynb`) in this format. `dummy_data/` holds an example:
   - `msbase_visits.csv` and `smsreg_visits.csv`: one row per visit with `patient_code, country, visit_date, birth_date, sex (F/M), diagnosis_date, disease_course (RR/SP), edss_score, edss_pyramidal, treatment_class`. `treatment_class` is one of `first_line, second_line, none, hsct, other`. SMSReg doesn't need `country`.
   - `msbase_relapses.csv` and `smsreg_relapses.csv`: `patient_code, relapse_date`. These are only used for the Lorscheider criteria.
2. Run `1_data_cleaning.ipynb`, then `2_icp_mondrian_shap.ipynb`. Everything is written to `custom_run/`, with models saved under the same names as above.
3. Run `3_analysis.ipynb` with `MODEL_OUTPUTS_DIR = "custom_run/model_outputs"`. Results go to `output/custom_run/`.

With the dummy data this runs as is. Hybrid models need at least two MSBase countries that meet the data-sufficiency criteria (> 1000 visits, ≥ 50 pwRRMS and ≥ 50 pwSPMS + pwMS-T).

## Folders

- `app/`: the Gradio app and its API (see [`app/README.md`](app/README.md)).
- `dummy_data/`: fake registry data (three training countries, two test-only countries and SMSReg), plus `make_dummy_data.py` to regenerate it. Every row has `DUMMY_DATA = 1`.
- `files/models/`: the published models (see above).
- `files/model_outputs/`: test-set p-values of all models (`*_results.json`), the Lorscheider comparison (`model_lor_cm.csv`, `lor_pred_visit_diff_results.json`) and `published_reference/` with numbers from the original analysis and the per-country summary used by the app.
- `files/chunks/`: the files over 50 MB, stored gzipped and split, because GitHub doesn't accept files over 100 MB. `pixi run merge` (`merge_files.py`) rebuilds them in `files/models/` and `files/model_outputs/` and checks their checksums. `python merge_files.py --split` re-creates the chunks.
- `harmonise_utils.py`: code shared by the notebooks and the app (features, model files, JSD, model assignment, Lorscheider criteria).
