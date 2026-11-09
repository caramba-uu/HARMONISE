# Synthetic patients for the app. Everything here is generated from random numbers; no registry data is used.
import numpy as np
import pandas as pd

# the simulated course is not returned: a synthetic patient has no recorded diagnosis
PROFILES = {
    "Non-progressing": "Low EDSS that goes up and down a little.",
    "Slowly progressing": "Slow EDSS increase over many years.",
    "Progressing": "A few stable visits, then a clear, sustained EDSS increase.",
}
VISIT_COLUMNS = ["visit_date", "edss_score", "edss_pyramidal", "treatment_class"]
TREATMENTS = ["first_line", "second_line", "none", "hsct", "other"]


def _step(value):
    return float(np.clip(round(value * 2) / 2, 0, 9.5))


def make_patient(profile, n_visits=10, seed=None):
    rng = np.random.default_rng(seed)
    n_visits = int(n_visits)
    birth = pd.Timestamp(int(rng.integers(1955, 1990)), int(rng.integers(1, 13)), 1)
    diagnosis = birth + pd.DateOffset(years=int(rng.integers(20, 40)), months=int(rng.integers(0, 12)))
    sex = str(rng.choice(["F", "M"], p=[0.7, 0.3]))

    if profile == "Non-progressing":
        switch, edss = None, float(rng.choice([1.0, 1.5, 2.0]))
    elif profile == "Slowly progressing":
        switch, edss = max(2, int(n_visits * 0.7)), float(rng.choice([2.0, 2.5]))
    else:
        switch, edss = max(2, int(n_visits * 0.4)), float(rng.choice([2.0, 2.5, 3.0]))

    rows, relapses = [], []
    date = diagnosis + pd.DateOffset(months=int(rng.integers(2, 12)))
    treatment = str(rng.choice(["first_line", "none"], p=[0.7, 0.3]))
    for v in range(n_visits):
        sp = switch is not None and v >= switch
        if profile == "Non-progressing":
            edss = _step(np.clip(edss + rng.choice([-0.5, 0, 0, 0.5]), 0, 3.0))
        elif profile == "Slowly progressing":
            edss = _step(edss + (rng.choice([0, 0.5, 0.5, 1.0]) if sp else rng.choice([0, 0, 0.5])))
        else:
            if v == switch:
                edss = _step(max(edss + 1.5, 4.5))
            else:
                edss = _step(edss + (rng.choice([0, 0.5, 1.0]) if sp else rng.choice([-0.5, 0, 0.5])))
        if sp and treatment == "first_line" and rng.random() < 0.5:
            treatment = "second_line"
        pyramidal = int(np.clip(round(edss / 2 + rng.choice([-1, 0, 1])), 0, 5))
        rows.append([date.strftime("%Y-%m-%d"), edss, pyramidal, treatment])
        if not sp and rng.random() < 0.25:
            relapses.append((date - pd.DateOffset(days=int(rng.integers(60, 150)))).strftime("%Y-%m-%d"))
        date += pd.DateOffset(months=int(rng.integers(8, 18)))

    return {
        "birth_date": birth.strftime("%Y-%m-%d"),
        "sex": sex,
        "diagnosis_date": diagnosis.strftime("%Y-%m-%d"),
        "visits": pd.DataFrame(rows, columns=VISIT_COLUMNS),
        "relapses": pd.DataFrame({"relapse_date": relapses}, dtype=str),
    }


def write_examples(out_dir):
    # the example/template CSVs in app/examples, in the same layout as dummy_data/*_visits.csv
    for i, profile in enumerate(PROFILES):
        p = make_patient(profile, 10, seed=i)
        name = profile.lower().replace(" ", "_").replace("-", "_")
        visits = p["visits"].assign(patient_code=f"DUMMY-{name}", birth_date=p["birth_date"], sex=p["sex"],
                                    diagnosis_date=p["diagnosis_date"], DUMMY_DATA=1)
        visits[["patient_code", "birth_date", "sex", "diagnosis_date"] + VISIT_COLUMNS + ["DUMMY_DATA"]].to_csv(
            f"{out_dir}/{name}_visits.csv", index=False)
        p["relapses"].assign(patient_code=f"DUMMY-{name}", DUMMY_DATA=1)[["patient_code", "relapse_date", "DUMMY_DATA"]].to_csv(
            f"{out_dir}/{name}_relapses.csv", index=False)


if __name__ == "__main__":
    # regenerate the examples: python app/dummy_patients.py
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    os.makedirs(out, exist_ok=True)
    write_examples(out)
