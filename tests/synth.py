"""Données synthétiques au format de diabetic_data.csv (uniquement pour les tests, pas de vraies données)."""
import numpy as np
import pandas as pd

MEDS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
        'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']


def raw_frame(n=3000, seed=0):
    r = np.random.RandomState(seed)
    d = pd.DataFrame({
        "encounter_id": np.arange(n) + 1000, "patient_nbr": np.arange(n) + 5000,
        "race": r.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"], n, p=[.6, .2, .05, .03, .05, .07]),
        "gender": r.choice(["Female", "Male", "Unknown/Invalid"], n, p=[.52, .479, .001]),
        "age": r.choice([f"[{a}-{a + 10})" for a in range(0, 100, 10)], n),
        "weight": r.choice(["?", "[75-100)", "[50-75)"], n, p=[.95, .03, .02]),
        "admission_type_id": r.choice([1, 2, 3, 5, 6], n, p=[.5, .2, .2, .05, .05]),
        "discharge_disposition_id": r.choice([1, 2, 3, 6, 7, 11, 18], n, p=[.6, .1, .1, .1, .03, .04, .03]),
        "admission_source_id": r.choice([1, 4, 7, 17], n, p=[.3, .05, .6, .05]),
        "time_in_hospital": r.randint(1, 15, n), "payer_code": r.choice(["MC", "HM", "SP", "?"], n),
        "medical_specialty": r.choice(["?", "InternalMedicine", "Cardiology", "Surgery-General", "Orthopedics-Reconstructive"], n),
        "num_lab_procedures": r.randint(1, 100, n), "num_procedures": r.randint(0, 7, n),
        "num_medications": r.randint(1, 40, n), "number_outpatient": r.poisson(.3, n),
        "number_emergency": r.poisson(.2, n), "number_inpatient": r.poisson(.4, n),
        "diag_1": r.choice(["250.83", "428", "414", "786", "V57", "E887", "?", "486", "577"], n),
        "diag_2": r.choice(["250.01", "401", "276", "?", "599"], n), "diag_3": r.choice(["250", "?", "272", "E849"], n),
        "number_diagnoses": r.randint(1, 10, n),
        "max_glu_serum": r.choice(["None", ">200", "Norm", ">300"], n, p=[.9, .03, .05, .02]),
        "A1Cresult": r.choice(["None", ">7", ">8", "Norm"], n, p=[.8, .05, .08, .07]),
        "change": r.choice(["No", "Ch"], n), "diabetesMed": r.choice(["Yes", "No"], n, p=[.75, .25]),
    })
    for m in MEDS:
        d[m] = "No" if m in ("examide", "citoglipton") else r.choice(["No", "Steady", "Up", "Down"], n, p=[.7, .22, .04, .04])
    score = 0.35 * d["number_inpatient"] + 0.02 * d["time_in_hospital"] + r.randn(n) * .8 - 2.6
    d["readmitted"] = np.where(score > np.quantile(score, 0.91), "<30", np.where(r.rand(n) < .3, ">30", "NO"))
    return d
