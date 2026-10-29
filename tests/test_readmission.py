"""Tests de la partie « Réadmission à 30 jours ». Lancer depuis le dossier du projet :  python -m unittest -v

Le modèle est un petit modèle factice entraîné sur des données synthétiques (même structure que celui de
train_readmission.py) : ces tests vérifient le code du site, pas la qualité du vrai modèle.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import app as site  # noqa: E402
import readmission_app as rd  # noqa: E402
import synth  # noqa: E402
from readmit_core import clean_raw, engineer  # noqa: E402


def fake_bundle():
    df, consts = engineer(clean_raw(synth.raw_frame(2500)))
    y, X = df["readmitted_30"], df.drop(columns="readmitted_30")
    num = [c for c in X.select_dtypes("number") if set(X[c].unique()) > {0, 1} or c not in ("change", "diabetesMed")]
    cats = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([("n", StandardScaler(), num),
                              ("c", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                                  sparse_output=False), cats)],
                             remainder="passthrough", verbose_feature_names_out=False)
    enc = pd.DataFrame(prep.fit_transform(X), columns=prep.get_feature_names_out(), index=X.index)
    model = LogisticRegression(class_weight="balanced", max_iter=500).fit(enc.values, y)
    proba = np.sort(model.predict_proba(enc.values)[:, 1])
    return {"model_name": "Logistic Regression", "preprocessor": prep, "selected_columns": list(enc.columns),
            "selected_variables": list(X.columns), "model": model, "threshold": 0.55,
            "top10_cutoff": float(np.quantile(proba, .9)), "reference_scores": proba,
            "input_columns": X.columns.tolist(), "constant_cols": consts,
            "categories": {c: sorted(X[c].astype(str).unique().tolist()) for c in cats},
            "metrics": {"test_roc_auc": .66, "test_pr_auc": .2, "recall": .4, "precision": .18, "top10_capture": .25}}


class ReadmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "model.joblib"
        joblib.dump(fake_bundle(), path)
        rd.load_model(path)
        cls.client = site.app.test_client()

    @classmethod
    def tearDownClass(cls):
        rd.load_model()           # remet le vrai fichier (ou l'erreur « introuvable »)
        cls.tmp.cleanup()

    def test_modele_charge(self):
        self.assertIsNone(rd.LOAD_ERROR, rd.LOAD_ERROR)

    def test_page_et_navigation(self):
        r = self.client.get("/readmission/")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("Estimer le risque", html)
        self.assertIn('href="/sejour/"', html)
        self.assertIn('href="/readmission/"', html)

    def test_defauts_donnent_une_prediction(self):
        r = self.client.post("/readmission/api/predict", json={})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        d = r.get_json()
        self.assertTrue(0 <= d["score"] <= 1)
        self.assertTrue(0 <= d["percentile"] <= 100)
        self.assertIn("niveau", d)

    def test_plus_d_hospitalisations_plus_de_risque(self):
        bas = self.client.post("/readmission/api/predict", json={"number_inpatient": 0}).get_json()["score"]
        haut = self.client.post("/readmission/api/predict", json={"number_inpatient": 15}).get_json()["score"]
        self.assertGreater(haut, bas)

    def test_valeurs_invalides(self):
        for payload in ({"time_in_hospital": 99}, {"num_medications": "abc"}, {"number_inpatient": -1},
                        {"gender": "inventé"}, {"age": "[999-1000)"}, {"time_in_hospital": "nan"}):
            r = self.client.post("/readmission/api/predict", json=payload)
            self.assertEqual(r.status_code, 422, payload)
            self.assertIn("erreur", r.get_json())

    def test_json_invalide(self):
        r = self.client.post("/readmission/api/predict", data="pas du json", content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_formulaire_html(self):
        r = self.client.post("/readmission/", data={"age": "[70-80)", "gender": "Male", "time_in_hospital": "5",
                                                    "num_lab_procedures": "40", "num_procedures": "1",
                                                    "num_medications": "12", "number_diagnoses": "7",
                                                    "number_outpatient": "0", "number_emergency": "1",
                                                    "number_inpatient": "2", "n_meds_used": "1", "n_meds_changed": "0",
                                                    "diabetesMed": "1"})
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("des patients ont un score plus faible", html)

    def test_formulaire_champ_vide_refuse(self):
        r = self.client.post("/readmission/", data={"age": "[70-80)", "time_in_hospital": ""})
        self.assertIn("obligatoire", r.get_data(as_text=True))

    def test_schema(self):
        d = self.client.get("/readmission/api/schema").get_json()
        self.assertIn("time_in_hospital", [f["name"] for f in d["numeriques"]])
        self.assertIn("gender", [f["name"] for f in d["categorielles"]])

    def test_health(self):
        h = self.client.get("/health").get_json()
        self.assertTrue(h["readmission"]["ok"])


class SansModele(unittest.TestCase):
    def test_message_clair_si_modele_absent(self):
        rd.load_model("/dossier/inexistant/model.joblib")
        try:
            c = site.app.test_client()
            self.assertEqual(c.get("/readmission/").status_code, 200)
            self.assertIn("introuvable", c.get("/readmission/").get_data(as_text=True))
            self.assertEqual(c.post("/readmission/api/predict", json={}).status_code, 503)
            self.assertEqual(c.get("/health").status_code, 200)
            self.assertFalse(c.get("/health").get_json()["readmission"]["ok"])
        finally:
            rd.load_model()


if __name__ == "__main__":
    unittest.main()
