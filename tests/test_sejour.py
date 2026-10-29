"""Tests de l'application. Lancer depuis le dossier du projet :  python -m unittest -v"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as site  # noqa: E402
import sejour_app as appmod  # noqa: E402


@unittest.skipIf(appmod.LOAD_ERROR, "Modèles absents : exportez-les depuis le notebook (section 8).")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = site.app.test_client()
        self.meta = appmod.META

    def valid_features(self, scenario):
        sc = self.meta["scenarios"][scenario]
        feats = {c: appmod.NUM_FIELDS.get(c, {}).get("default", 0) for c in sc["numeriques"]}
        feats["age_num"] = 65
        for c in sc["categorielles"]:
            feats[c] = self.meta["options"][c][0]
        return feats

    def post(self, payload):
        return self.client.post("/sejour/api/predict", data=json.dumps(payload), content_type="application/json")

    def test_pages(self):
        self.assertEqual(self.client.get("/sejour/").status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/sejour/api/schema").status_code, 200)
        self.assertEqual(self.client.get("/nimporte-quoi").status_code, 404)

    def test_prediction_valide_pour_chaque_scenario(self):
        for sc in self.meta["scenarios"]:
            r = self.post({"scenario": sc, "features": self.valid_features(sc)})
            self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
            d = r.get_json()
            self.assertTrue(1 <= d["jours"] <= 14)
            self.assertIn(d["categorie"], {"court", "moyen", "long"})
            self.assertLessEqual(d["intervalle"][0], d["jours"])
            self.assertGreaterEqual(d["intervalle"][1], d["jours"])

    def test_scenario_en_minuscule_accepte(self):
        r = self.post({"scenario": "a", "features": self.valid_features("A")})
        self.assertEqual(r.status_code, 200)

    def test_champ_manquant(self):
        f = self.valid_features("A"); f.pop("gender")
        r = self.post({"scenario": "A", "features": f})
        self.assertEqual(r.status_code, 422)
        self.assertIn("gender", r.get_json()["champs"])

    def test_valeurs_invalides(self):
        f = self.valid_features("A")
        f["number_inpatient"] = -3
        f["number_emergency"] = "abc"
        f["race"] = "valeur-inventee"
        r = self.post({"scenario": "A", "features": f})
        self.assertEqual(r.status_code, 422)
        champs = r.get_json()["champs"]
        for c in ("number_inpatient", "number_emergency", "race"):
            self.assertIn(c, champs)

    def test_valeur_extreme_plafonnee(self):
        f = self.valid_features("A"); f["number_inpatient"] = 10_000
        r = self.post({"scenario": "A", "features": f})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["avertissements"])

    def test_nan_et_infini_refuses(self):
        f = self.valid_features("A"); f["number_inpatient"] = "nan"
        self.assertEqual(self.post({"scenario": "A", "features": f}).status_code, 422)
        f["number_inpatient"] = "inf"
        self.assertEqual(self.post({"scenario": "A", "features": f}).status_code, 422)

    def test_requetes_mal_formees(self):
        self.assertEqual(self.client.post("/sejour/api/predict", data="pas du json",
                                          content_type="application/json").status_code, 400)
        self.assertEqual(self.post({"scenario": "Z", "features": {}}).status_code, 400)
        self.assertEqual(self.post({"scenario": "A", "features": "x"}).status_code, 400)
        self.assertEqual(self.client.get("/sejour/api/predict").status_code, 405)

    def test_requete_trop_grosse(self):
        r = self.post({"scenario": "A", "features": {"x": "a" * 40_000}})
        self.assertEqual(r.status_code, 413)


class SansModeles(unittest.TestCase):
    def test_message_clair_si_modeles_absents(self):
        old = (appmod.META, appmod.PIPES, appmod.LOAD_ERROR, appmod.MODEL_DIR)
        try:
            appmod.MODEL_DIR = Path("/dossier/inexistant")
            appmod.load_models()
            c = site.app.test_client()
            self.assertEqual(c.get("/health").status_code, 200)          # le site reste en ligne
            self.assertEqual(c.get("/health").get_json()["statut"], "partiel")
            self.assertEqual(c.get("/sejour/").status_code, 503)
            self.assertIn("introuvables", c.get("/sejour/").get_data(as_text=True))
        finally:
            appmod.META, appmod.PIPES, appmod.LOAD_ERROR, appmod.MODEL_DIR = old


if __name__ == "__main__":
    unittest.main()
