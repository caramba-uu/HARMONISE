# Generates the fake registry data in this folder (run from the repository root).
# Every row has DUMMY_DATA = 1 so the notebooks can warn when dummy data is used.
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)

# AU, CA and IT are big enough to be training countries, AR and JP are test only
MSBASE = {"AU": 300, "CA": 300, "IT": 300, "AR": 60, "JP": 60}


def make_patients(n_per_country, prefix):
    visits, relapses = [], []
    for cc, n in n_per_country.items():
        for k in range(n):
            pid = f"DUMMY-{prefix}-{cc}-{k:04d}"
            course = rng.choice(["RR", "SP", "T"], p=[0.6, 0.2, 0.2])  # T = transitions from RR to SP
            birth = pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.integers(0, 365 * 40)))
            diagnosis = birth + pd.Timedelta(days=int(rng.integers(365 * 20, 365 * 40)))
            n_visits = int(rng.integers(4, 13))
            switch = int(rng.integers(2, n_visits)) if course == "T" else None
            t = diagnosis + pd.Timedelta(days=int(rng.integers(30, 400)))
            edss = float(rng.choice([4.0, 4.5, 5.0, 6.0])) if course == "SP" else float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0]))
            sex = rng.choice(["F", "M"], p=[0.7, 0.3])

            for v in range(n_visits):
                sp = course == "SP" or (course == "T" and v >= switch)
                if sp:
                    edss = min(9.0, max(edss, 3.5) + float(rng.choice([0, 0.5, 1.0])))
                else:
                    edss = float(np.clip(edss + rng.choice([-0.5, 0, 0.5]), 0, 6))
                visits.append({
                    "patient_code": pid, "country": cc, "visit_date": t.date(), "birth_date": birth.date(),
                    "sex": sex, "diagnosis_date": diagnosis.date(), "disease_course": "SP" if sp else "RR",
                    "edss_score": edss, "edss_pyramidal": rng.choice([np.nan, 1, 2, 3]),
                    "treatment_class": rng.choice(["first_line", "second_line", "none", "hsct", "other"],
                                                  p=[0.45, 0.25, 0.2, 0.02, 0.08]),
                    "DUMMY_DATA": 1,
                })
                if rng.random() < 0.15:
                    relapse = t - pd.Timedelta(days=int(rng.integers(0, 60)))
                    relapses.append({"patient_code": pid, "relapse_date": relapse.date(), "DUMMY_DATA": 1})
                t += pd.Timedelta(days=int(rng.integers(150, 500)))
    return pd.DataFrame(visits), pd.DataFrame(relapses)


visits, relapses = make_patients(MSBASE, "MS")
visits.to_csv("dummy_data/msbase_visits.csv", index=False)
relapses.to_csv("dummy_data/msbase_relapses.csv", index=False)

visits, relapses = make_patients({"SE": 400}, "SE")
visits.drop(columns="country").to_csv("dummy_data/smsreg_visits.csv", index=False)
relapses.to_csv("dummy_data/smsreg_relapses.csv", index=False)
