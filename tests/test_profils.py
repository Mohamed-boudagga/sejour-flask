"""Tests du module « Profils de santé » (clustering CDC, K-Means) monté sous /profils."""
import unittest

import app as site
import profils_app

VALID = {"HighBP": "1", "HighChol": "0", "CholCheck": "1", "BMI": "27", "Smoker": "0", "Stroke": "0",
         "HeartDiseaseorAttack": "0", "PhysActivity": "1", "Fruits": "1", "Veggies": "1", "HvyAlcoholConsump": "0",
         "AnyHealthcare": "1", "NoDocbcCost": "0", "GenHlth": "3", "MentHlth": "2", "PhysHlth": "4", "DiffWalk": "0",
         "Sex": "1", "Age": "8", "Education": "5", "Income": "6"}


class ProfilsTest(unittest.TestCase):
    def setUp(self):
        self.client = site.app.test_client()

    def test_modele_charge(self):
        self.assertIsNone(profils_app.READY_ERROR, profils_app.READY_ERROR)
        self.assertEqual(profils_app.ARTIFACT["best_model_name"], "K-Means")

    def test_page_get(self):
        r = self.client.get("/profils/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Profils de santé".encode(), r.data)       # navigation commune

    def test_formulaire_donne_un_cluster(self):
        r = self.client.post("/profils/", data=VALID)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Cluster", r.data)

    def test_formulaire_champ_manquant(self):
        bad = dict(VALID); bad["BMI"] = ""
        r = self.client.post("/profils/", data=bad)
        self.assertEqual(r.status_code, 200)
        self.assertIn("obligatoire".encode(), r.data)

    def test_api_ok(self):
        r = self.client.post("/profils/api/predict", json={k: float(v) for k, v in VALID.items()})
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertIn("cluster_id", j)
        self.assertEqual(j["model_name"], "K-Means")

    def test_api_valeur_hors_plage(self):
        r = self.client.post("/profils/api/predict", json={"BMI": 500})
        self.assertEqual(r.status_code, 400)

    def test_api_variable_inconnue(self):
        r = self.client.post("/profils/api/predict", json={"nimporte": 1})
        self.assertEqual(r.status_code, 400)

    def test_api_corps_invalide(self):
        r = self.client.post("/profils/api/predict", data="pas du json", content_type="text/plain")
        self.assertEqual(r.status_code, 400)

    def test_health_inclut_profils(self):
        j = self.client.get("/health").get_json()
        self.assertTrue(j["profils"]["ok"])


if __name__ == "__main__":
    unittest.main()
