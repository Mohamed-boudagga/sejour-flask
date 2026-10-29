"""Partie « Durée de séjour » (scénarios A et B) : blueprint Flask monté sous /sejour."""
import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Blueprint, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "sejour"

bp = Blueprint("sejour", __name__, url_prefix="/sejour")
log = logging.getLogger("sejour")

# --------------------------------------------------------------------------
# Libellés lisibles pour le formulaire (les valeurs brutes restent celles du modèle)
# --------------------------------------------------------------------------
AGES = {5: "0-10 ans", 15: "10-20 ans", 25: "20-30 ans", 35: "30-40 ans", 45: "40-50 ans",
        55: "50-60 ans", 65: "60-70 ans", 75: "70-80 ans", 85: "80-90 ans", 95: "90-100 ans"}

NUM_FIELDS = {
    "age_num":            {"label": "Tranche d'âge", "help": "Âge du patient à l'admission."},
    "number_outpatient":  {"label": "Consultations externes", "help": "Nombre de consultations externes dans l'année précédente.", "default": 0},
    "number_emergency":   {"label": "Passages aux urgences", "help": "Nombre de passages aux urgences dans l'année précédente.", "default": 0},
    "number_inpatient":   {"label": "Hospitalisations précédentes", "help": "Nombre d'hospitalisations dans l'année précédente.", "default": 0},
    "num_lab_procedures": {"label": "Examens de laboratoire", "help": "Nombre d'examens de laboratoire réalisés pendant le séjour.", "default": 40},
    "num_procedures":     {"label": "Procédures (hors laboratoire)", "help": "Nombre de procédures réalisées pendant le séjour.", "default": 1},
    "num_medications":    {"label": "Médicaments administrés", "help": "Nombre de médicaments distincts administrés pendant le séjour.", "default": 15},
    "number_diagnoses":   {"label": "Nombre de diagnostics", "help": "Nombre de diagnostics enregistrés.", "default": 7},
    "n_meds_active":      {"label": "Nombre d'antidiabétiques prescrits", "help": "Nombre de médicaments antidiabétiques prescrits.", "default": 1},
    "n_meds_changed":     {"label": "Dont dose modifiée", "help": "Parmi eux, nombre dont la dose a été augmentée ou diminuée.", "default": 0},
}

CAT_FIELDS = {
    "gender":              "Sexe",
    "race":                "Origine",
    "admission_type_id":   "Type d'admission",
    "admission_source_id": "Source d'admission",
    "payer_code":          "Assurance (code payeur)",
    "medical_specialty":   "Spécialité du médecin admetteur",
    "diag_1_group":        "Diagnostic principal",
    "A1Cresult":           "Résultat HbA1c",
    "max_glu_serum":       "Glycémie sérique maximale",
    "change":              "Changement de traitement",
    "diabetesMed":         "Un antidiabétique a-t-il été prescrit ?",
}

VALUE_LABELS = {
    "gender": {"Male": "Homme", "Female": "Femme"},
    "race": {"Caucasian": "Caucasien", "AfricanAmerican": "Afro-américain", "Hispanic": "Hispanique",
             "Asian": "Asiatique", "Other": "Autre", "Unknown": "Non renseigné"},
    "admission_type_id": {"1": "Urgence", "2": "Urgence différée", "3": "Programmée", "4": "Nouveau-né",
                          "5": "Non disponible", "6": "Non renseigné", "7": "Centre de traumatologie",
                          "8": "Non attribué", "Other": "Autre"},
    "admission_source_id": {"1": "Orientation par un médecin", "2": "Orientation par une clinique",
                            "3": "Orientation par un HMO", "4": "Transfert depuis un hôpital",
                            "5": "Transfert depuis un établissement de soins infirmiers",
                            "6": "Transfert depuis un autre établissement", "7": "Urgences",
                            "8": "Justice / police", "9": "Non disponible", "17": "Non renseigné",
                            "20": "Non attribué", "10": "Transfert depuis un hôpital d'accès critique", "11": "Accouchement normal",
                            "14": "Naissance hors établissement",
                            "22": "Transfert interne", "25": "Transfert depuis chirurgie ambulatoire",
                            "Other": "Autre"},
    "payer_code": {"MC": "Medicare", "MD": "Medicaid", "HM": "HMO", "SP": "Paiement personnel",
                   "BC": "Blue Cross", "Unknown": "Non renseigné"},
    "medical_specialty": {"Unknown": "Non renseignée"},
    "diag_1_group": {"Diabetes": "Diabète", "Circulatory": "Cardio-vasculaire", "Respiratory": "Respiratoire",
                     "Digestive": "Digestif", "Genitourinary": "Génito-urinaire",
                     "Musculoskeletal": "Musculo-squelettique", "Injury": "Traumatisme",
                     "Neoplasms": "Tumeurs", "Other": "Autre", "Unknown": "Non renseigné"},
    "A1Cresult": {"NotTested": "Non réalisé", "Norm": "Normal", ">7": "Supérieur à 7 %", ">8": "Supérieur à 8 %"},
    "max_glu_serum": {"NotTested": "Non réalisée", "Norm": "Normale", ">200": "Supérieure à 200 mg/dL",
                      ">300": "Supérieure à 300 mg/dL"},
    "change": {"No": "Aucun changement", "Ch": "Traitement modifié"},
    "diabetesMed": {"Yes": "Oui", "No": "Non"},
}

SCENARIO_TEXT = {
    "A": {"titre": "À l'admission",
          "sous_titre": "Anticiper la durée du séjour dès l'arrivée du patient.",
          "usage": "Utilise uniquement des informations connues à l'admission."},
    "B": {"titre": "En cours de séjour",
          "sous_titre": "Réévaluer la durée prévue avec les examens et traitements déjà réalisés.",
          "usage": "Plus précis, mais utilise des informations qui s'accumulent pendant le séjour."},
}


def pretty(value: str) -> str:
    """'InternalMedicine' -> 'Internal Medicine' (valeurs sans traduction)."""
    out = []
    for i, ch in enumerate(value):
        if i and ch.isupper() and value[i - 1].islower():
            out.append(" ")
        out.append(ch)
    return "".join(out).replace("_", " ").replace("-", " - ")


def value_label(col: str, value: str) -> str:
    label = VALUE_LABELS.get(col, {}).get(value)
    if label:
        return label
    if col == "payer_code":                      # codes d'assurance sans libellé officiel connu
        return f"Code {value}"
    return pretty(value)


# --------------------------------------------------------------------------
# Chargement des modèles
# --------------------------------------------------------------------------
META = None
PIPES = {}
LOAD_ERROR = None


def load_models():
    global META, PIPES, LOAD_ERROR
    try:
        with open(MODEL_DIR / "model_meta.json", encoding="utf-8") as f:
            META = json.load(f)
        PIPES = {k: joblib.load(MODEL_DIR / f"model_{k}.joblib") for k in META["scenarios"]}
        if META.get("sklearn_version") != sklearn.__version__:
            log.warning("Version de scikit-learn différente : modèle entraîné avec %s, installée %s",
                        META.get("sklearn_version"), sklearn.__version__)
        LOAD_ERROR = None
        log.info("Modèles chargés : %s", ", ".join(PIPES))
    except FileNotFoundError as exc:
        META, PIPES = None, {}
        LOAD_ERROR = ("Fichiers du modèle introuvables dans le dossier « models/sejour/ » "
                      f"({Path(exc.filename).name if exc.filename else 'fichier manquant'}). "
                      "Exportez-les depuis le notebook (section 8) puis copiez-les dans ce dossier.")
    except Exception as exc:  # noqa: BLE001 - on veut afficher n'importe quelle cause
        META, PIPES = None, {}
        LOAD_ERROR = f"Impossible de charger les modèles : {exc}"
    if LOAD_ERROR:
        log.error(LOAD_ERROR)


load_models()


# --------------------------------------------------------------------------
# Schéma du formulaire
# --------------------------------------------------------------------------
def build_schema():
    schema = {}
    for key, sc in META["scenarios"].items():
        fields = []
        for col in sc["numeriques"]:
            info = NUM_FIELDS.get(col, {"label": col})
            if col == "age_num":
                fields.append({"name": col, "label": info["label"], "help": info.get("help"), "type": "select",
                               "default": "65",
                               "options": [{"value": str(v), "label": l} for v, l in AGES.items()]})
            else:
                cap = META.get("plafonds", {}).get(col)
                fields.append({"name": col, "label": info["label"], "help": info.get("help"), "type": "number",
                               "min": 0, "max": int(math.ceil(cap)) if cap is not None else 999,
                               "default": info.get("default", 0)})
        for col in sc["categorielles"]:
            opts = META["options"].get(col, [])
            fields.append({"name": col, "label": CAT_FIELDS.get(col, col), "type": "select",
                           "options": [{"value": v, "label": value_label(col, v)} for v in opts]})
        schema[key] = {"fields": fields, **SCENARIO_TEXT.get(key, {}),
                       "modele": sc["modele"].split(" - ", 1)[-1],
                       "performances": sc["performances"]}
    return schema


# --------------------------------------------------------------------------
# Prédiction
# --------------------------------------------------------------------------
class ValidationError(Exception):
    def __init__(self, errors):
        super().__init__("Données invalides")
        self.errors = errors


def reliability(r2: float) -> dict:
    if r2 < 0.2:
        return {"niveau": "faible", "texte": "Fiabilité faible : le modèle explique peu la durée de séjour. "
                                              "Utilisez cette estimation comme un ordre de grandeur."}
    if r2 < 0.5:
        return {"niveau": "modérée", "texte": "Fiabilité modérée : l'estimation donne une tendance, "
                                               "avec une marge d'erreur à prendre en compte."}
    return {"niveau": "bonne", "texte": "Fiabilité correcte pour une estimation de planification."}


def predict(scenario: str, values: dict) -> dict:
    sc = META["scenarios"][scenario]
    errors, row, warnings = {}, {}, []

    for col in sc["numeriques"]:
        raw = values.get(col)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors[col] = "Entrez un nombre."
            continue
        if not math.isfinite(v) or v < 0:
            errors[col] = "Entrez un nombre positif ou nul."
            continue
        cap = META.get("plafonds", {}).get(col)
        if cap is not None and v > cap:
            warnings.append(f"« {NUM_FIELDS.get(col, {}).get('label', col)} » dépasse la plage usuelle "
                            f"({cap:g}) : la valeur a été plafonnée, comme à l'entraînement.")
            v = cap
        row[col] = v

    for col in sc["categorielles"]:
        raw = values.get(col)
        if raw is None or str(raw) == "":
            errors[col] = "Faites un choix."
        elif str(raw) not in META["options"].get(col, []):
            errors[col] = "Valeur non reconnue."
        else:
            row[col] = str(raw)

    if errors:
        raise ValidationError(errors)

    cols = sc["numeriques"] + sc["categorielles"]
    X = pd.DataFrame([row])[cols]
    pred = float(np.clip(PIPES[scenario].predict(X)[0], 1, 14))
    perf = sc["performances"]
    mae = perf["mae"]
    rounded = int(round(pred))
    categorie = "court" if rounded <= 3 else ("moyen" if rounded <= 6 else "long")
    return {
        "scenario": scenario,
        "modele": sc["modele"],
        "jours": round(pred, 1),
        "intervalle": [round(max(1.0, pred - mae), 1), round(min(14.0, pred + mae), 1)],
        "marge": round(mae, 1),
        "categorie": categorie,
        "fiabilite": {**reliability(perf["r2"]), "r2": round(perf["r2"], 2)},
        "avertissements": warnings,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@bp.get("/")
def index():
    if LOAD_ERROR:
        return render_template("sejour.html", error=LOAD_ERROR, schema=None, meta=None), 503
    return render_template("sejour.html", error=None, schema=build_schema(), meta=META)


@bp.get("/api/schema")
def api_schema():
    if LOAD_ERROR:
        return jsonify(erreur=LOAD_ERROR), 503
    return jsonify(build_schema())


@bp.post("/api/predict")
def api_predict():
    if LOAD_ERROR:
        return jsonify(erreur=LOAD_ERROR), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(erreur="Corps JSON attendu : {\"scenario\": \"A\", \"features\": {...}}"), 400
    scenario = str(payload.get("scenario", "")).upper()
    if scenario not in PIPES:
        return jsonify(erreur=f"Scénario inconnu. Valeurs possibles : {sorted(PIPES)}"), 400
    features = payload.get("features")
    if not isinstance(features, dict):
        return jsonify(erreur="Le champ « features » doit être un objet JSON."), 400
    try:
        return jsonify(predict(scenario, features))
    except ValidationError as exc:
        return jsonify(erreur="Données invalides", champs=exc.errors), 422
