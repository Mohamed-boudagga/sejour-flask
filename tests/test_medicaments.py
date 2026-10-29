"""Tests de la partie « Nombre de médicaments ». Lancer depuis le dossier du projet :  python -m unittest -v

Les tests construisent un petit modèle factice (même structure que l'artefact du notebook, cellule 4.6) : ils
vérifient le code du site, pas la qualité du vrai modèle. Un dernier test s'applique au vrai fichier s'il existe.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as appmod  # noqa: E402
import medicaments_app as med  # noqa: E402

NUM = ["time_in_hospital", "num_procedures", "num_lab_procedures", "number_diagnoses", "age"]
CAT = {"admission_type_id": ["1", "2", "3", "Inconnu"], "medical_specialty": ["Inconnu", "Cardiology", "InternalMedicine"],
       "diag1_group": ["Diabete", "Circulatoire", "Autre"]}
CAPS = {"time_in_hospital": 12.0, "num_procedures": 6.0, "num_lab_procedures": 90.0, "number_diagnoses": 9.0, "age": 95.0}


def fake_artefact():
    rng = np.random.RandomState(0)
    n = 600
    X = pd.DataFrame({c: rng.randint(1, int(CAPS[c]), n) for c in NUM})
    for c, vals in CAT.items():
        X[c] = rng.choice(vals, n)
    y = 5 + 0.8 * X["time_in_hospital"] + 0.1 * X["num_lab_procedures"] + rng.randn(n)
    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False), list(CAT))])
    pipe = Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(max_iter=30, random_state=0))]).fit(X, y)
    return {"modele": pipe, "nom_modele": "LightGBM", "variables": NUM + list(CAT), "variables_numeriques": NUM,
            "variables_categorielles": list(CAT), "plafonds": CAPS, "valeurs_possibles": CAT,
            "metriques_test": {"MAE": 4.43, "RMSE": 5.76, "R² test": 0.499, "R² entraînement": 0.524, "% à ±3 médicaments": 42.7}}


class MedicamentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "modele_medicaments.joblib"
        joblib.dump(fake_artefact(), path)
        med.load_model(path)
        cls.client = appmod.app.test_client()

    @classmethod
    def tearDownClass(cls):
        med.load_model()          # remet le vrai fichier (ou l'erreur « introuvable »)
        cls.tmp.cleanup()

    def valid(self):
        d = {c: 5 for c in NUM}
        d["age"] = 65
        d.update({c: v[0] for c, v in CAT.items()})
        return d

    def post(self, payload):
        return self.client.post("/medicaments/api/predict", data=json.dumps(payload), content_type="application/json")

    def test_modele_charge_et_pages(self):
        self.assertIsNone(med.LOAD_ERROR)
        for url in ("/", "/medicaments/", "/medicaments/api/schema", "/health"):
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_prediction_valide(self):
        r = self.post(self.valid())
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        d = r.get_json()
        self.assertGreaterEqual(d["medicaments"], 0)
        self.assertLessEqual(d["intervalle"][0], d["arrondi"])
        self.assertGreaterEqual(d["intervalle"][1], d["arrondi"])

    def test_plus_de_sejour_plus_de_medicaments(self):
        a, b = self.valid(), self.valid()
        a["time_in_hospital"], b["time_in_hospital"] = 2, 12
        self.assertGreater(self.post(b).get_json()["medicaments"], self.post(a).get_json()["medicaments"])

    def test_valeur_extreme_plafonnee(self):
        d = self.valid(); d["num_lab_procedures"] = 10_000
        r = self.post(d)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["avertissements"])

    def test_virgule_decimale_acceptee(self):
        d = self.valid(); d["time_in_hospital"] = "3,0"
        self.assertEqual(self.post(d).status_code, 200)

    def test_entrees_invalides(self):
        for champ, valeur in [("time_in_hospital", -3), ("time_in_hospital", "abc"), ("time_in_hospital", "nan"),
                              ("time_in_hospital", "inf"), ("time_in_hospital", ""), ("age", 200),
                              ("diag1_group", "valeur-inventee")]:
            d = self.valid(); d[champ] = valeur
            self.assertEqual(self.post(d).status_code, 422, (champ, valeur))

    def test_champ_manquant(self):
        d = self.valid(); d.pop("num_procedures")
        self.assertEqual(self.post(d).status_code, 422)

    def test_corps_mal_forme(self):
        r = self.client.post("/medicaments/api/predict", data="pas du json", content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_formulaire_html(self):
        form = {k: str(v) for k, v in self.valid().items()}
        html = self.client.post("/medicaments/", data=form).get_data(as_text=True)
        self.assertIn("médicaments distincts", html)
        html = self.client.post("/medicaments/", data={**form, "time_in_hospital": "-1"}).get_data(as_text=True)
        self.assertIn("hors plage", html)


class MessageSansModele(unittest.TestCase):
    def test_fichier_absent(self):
        med.load_model(Path(tempfile.gettempdir()) / "nexiste-pas.joblib")
        try:
            c = appmod.app.test_client()
            self.assertEqual(c.get("/medicaments/").status_code, 200)
            self.assertIn("introuvable", c.get("/medicaments/").get_data(as_text=True))
            self.assertEqual(c.post("/medicaments/api/predict", json={}).status_code, 503)
            self.assertEqual(c.get("/health").status_code, 200)   # le reste du site continue de répondre
            self.assertEqual(c.get("/issue/").status_code, 200)
        finally:
            med.load_model()


@unittest.skipIf(not med.MODEL_PATH.exists(), "Vrai modèle absent de models/medicaments/")
class VraiModele(unittest.TestCase):
    def test_prediction_plausible(self):
        med.load_model()
        self.assertIsNone(med.LOAD_ERROR, med.LOAD_ERROR)
        d = {f["name"]: (65 if f["kind"] == "age" else 3) for f in med.NUM_FIELDS}
        d.update({f["name"]: f["default"] for f in med.CAT_FIELDS})
        r = appmod.app.test_client().post("/medicaments/api/predict", json=d)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        self.assertTrue(0 <= r.get_json()["medicaments"] <= 100)


if __name__ == "__main__":
    unittest.main()
