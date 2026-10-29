"""Partie « Nombre de médicaments » (LightGBM, régression) : blueprint Flask monté sous /medicaments.

Le fichier models/medicaments/modele_medicaments.joblib est produit par la cellule 4.6 du notebook. C'est un
dictionnaire : modele (préparation + modèle), variables, variables_numeriques, variables_categorielles, plafonds,
valeurs_possibles, metriques_test. Le formulaire est construit à partir de ce dictionnaire : si les 7 variables
changent après un nouvel entraînement, il suffit de remplacer le fichier.
"""
import logging
import math
from pathlib import Path

import compat  # noqa: F401  (doit précéder joblib.load)
import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, render_template, request

import sejour_app

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "medicaments" / "modele_medicaments.joblib"
bp = Blueprint("medicaments", __name__, url_prefix="/medicaments")
log = logging.getLogger("medicaments")

# --------------------------------------------------------------------------
# Libellés lisibles (les valeurs brutes restent celles du modèle)
# --------------------------------------------------------------------------
NUM_LABELS = {
    "time_in_hospital": ("Durée du séjour (jours)", "Connue seulement à la sortie : l'estimation porte sur un séjour presque terminé."),
    "num_lab_procedures": ("Examens de laboratoire", "Nombre d'examens de laboratoire pendant le séjour."),
    "num_procedures": ("Procédures (hors laboratoire)", "Nombre de procédures pendant le séjour."),
    "number_diagnoses": ("Nombre de diagnostics", "Nombre de diagnostics enregistrés."),
    "number_outpatient": ("Consultations externes", "Dans l'année précédente."),
    "number_emergency": ("Passages aux urgences", "Dans l'année précédente."),
    "number_inpatient": ("Hospitalisations précédentes", "Dans l'année précédente."),
}
CAT_LABELS = {
    "gender": "Sexe", "race": "Origine", "admission_type_id": "Type d'admission",
    "admission_source_id": "Source d'admission", "discharge_disposition_id": "Type de sortie",
    "medical_specialty": "Spécialité médicale", "diag1_group": "Diagnostic principal",
    "diag2_group": "Diagnostic secondaire 1", "diag3_group": "Diagnostic secondaire 2",
}
DIAG_VALUES = {"Diabete": "Diabète", "Circulatoire": "Cardio-vasculaire", "Respiratoire": "Respiratoire",
               "Digestif": "Digestif", "Traumatisme": "Traumatisme", "Musculo": "Musculo-squelettique",
               "Genito-urinaire": "Génito-urinaire", "Tumeurs": "Tumeurs", "Autre": "Autre", "Inconnu": "Non renseigné"}
DISCHARGE = {"1": "Domicile", "2": "Transfert vers un hôpital de soins aigus", "3": "Établissement de soins infirmiers",
             "4": "Établissement de soins intermédiaires", "5": "Autre établissement de soins",
             "6": "Domicile avec soins à domicile", "7": "Sortie contre avis médical", "8": "Domicile avec perfusions",
             "9": "Admis comme hospitalisé", "11": "Décès", "13": "Soins palliatifs (domicile)",
             "14": "Soins palliatifs (établissement)", "22": "Transfert en rééducation", "23": "Soins de longue durée",
             "Inconnu": "Non renseigné"}
GENDER = {"Male": "Homme", "Female": "Femme", "Unknown/Invalid": "Non renseigné"}
AGE_LABELS = sejour_app.AGES  # milieu de tranche (5, 15, ...) -> « 0-10 ans »


def value_label(col, value):
    if col.startswith("diag"):
        return DIAG_VALUES.get(value, value)
    if col == "discharge_disposition_id":
        return DISCHARGE.get(value, "Non renseigné" if value == "Inconnu" else f"Code {value}")
    if col == "gender":
        return GENDER.get(value, value)
    if col in ("admission_type_id", "admission_source_id", "race"):
        return "Non renseigné" if value == "Inconnu" else sejour_app.value_label(col, value)
    if value in ("Inconnu", "Unknown"):
        return "Non renseignée"
    return sejour_app.pretty(value)


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
ART = None
LOAD_ERROR = None
NUM_FIELDS, CAT_FIELDS = [], []


def load_model(path=None):
    """Charge l'artefact et prépare les champs du formulaire. Ne lève jamais : renseigne LOAD_ERROR."""
    global ART, LOAD_ERROR, NUM_FIELDS, CAT_FIELDS
    path = Path(path) if path else MODEL_PATH
    try:
        art = joblib.load(path)
        for k in ("modele", "variables", "variables_numeriques", "variables_categorielles", "plafonds", "valeurs_possibles"):
            if k not in art:
                raise KeyError(f"clé « {k} » absente de l'artefact")
        num, fields_num = art["variables_numeriques"], []
        for c in num:
            if c == "age":
                fields_num.append({"name": c, "label": "Tranche d'âge", "kind": "age",
                                   "options": [(str(k), v) for k, v in AGE_LABELS.items()], "help": ""})
                continue
            label, hlp = NUM_LABELS.get(c, (c, ""))
            cap = art["plafonds"].get(c)
            fields_num.append({"name": c, "label": label, "kind": "number", "help": hlp,
                               "max": None if cap is None else math.ceil(cap)})
        fields_cat = []
        for c in art["variables_categorielles"]:
            vals = [str(v) for v in art["valeurs_possibles"][c]]
            default = next((v for v in ("Inconnu", "Unknown") if v in vals), vals[0])
            fields_cat.append({"name": c, "label": CAT_LABELS.get(c, c), "default": default,
                               "options": [(v, value_label(c, v)) for v in vals]})
        ART, NUM_FIELDS, CAT_FIELDS, LOAD_ERROR = art, fields_num, fields_cat, None
        log.info("Modèle médicaments chargé : %s", art.get("nom_modele", "?"))
    except FileNotFoundError:
        ART, NUM_FIELDS, CAT_FIELDS = None, [], []
        LOAD_ERROR = ("Fichier du modèle introuvable : copiez « modele_medicaments.joblib » (cellule 4.6 du notebook) "
                      "dans le dossier « models/medicaments/ », puis redémarrez le site.")
    except Exception as exc:  # noqa: BLE001 - afficher n'importe quelle cause
        ART, NUM_FIELDS, CAT_FIELDS = None, [], []
        LOAD_ERROR = f"Impossible de charger le modèle : {exc}"
    if LOAD_ERROR:
        log.error(LOAD_ERROR)


load_model()


# --------------------------------------------------------------------------
# Prédiction
# --------------------------------------------------------------------------
def _num(v, label):
    if v is None or str(v).strip() == "":
        raise ValueError(f"« {label} » est obligatoire.")
    try:
        x = float(str(v).replace(",", "."))
    except ValueError:
        raise ValueError(f"« {label} » : valeur numérique invalide (« {v} »).")
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"« {label} » : valeur hors plage (« {v} »).")
    return x


def predict(form):
    """form : dict (formulaire ou JSON). Retourne un dict ; lève ValueError si l'entrée est invalide."""
    if LOAD_ERROR:
        raise RuntimeError(LOAD_ERROR)
    row, warnings = {}, []
    for f in NUM_FIELDS:
        c, x = f["name"], _num(form.get(f["name"]), f["label"])
        if c == "age" and x > 120:
            raise ValueError("Âge hors plage (0–120).")
        cap = ART["plafonds"].get(c)
        if cap is not None and x > cap:       # même plafonnement que l'entraînement (99e percentile)
            warnings.append(f"« {f['label']} » ramené à {cap:g} (valeur maximale vue à l'entraînement).")
            x = cap
        row[c] = x
    for f in CAT_FIELDS:
        v = str(form.get(f["name"], f["default"]))
        if v not in {o[0] for o in f["options"]}:
            raise ValueError(f"« {f['label']} » : valeur inconnue (« {v} »).")
        row[f["name"]] = v
    X = pd.DataFrame([row])[ART["variables"]]
    y = max(float(ART["modele"].predict(X)[0]), 0.0)
    mae = float(ART.get("metriques_test", {}).get("MAE", 0)) or None
    out = {"medicaments": round(y, 1), "arrondi": int(round(y)), "avertissements": warnings}
    if mae:
        out["erreur_moyenne"] = round(mae, 1)
        out["intervalle"] = [max(0, int(round(y - mae))), int(round(y + mae))]
    return out


def metrics():
    m = (ART or {}).get("metriques_test", {})
    g = lambda k: m.get(k)  # noqa: E731
    return {"nom": (ART or {}).get("nom_modele", "LightGBM"), "r2": g("R² test"), "mae": g("MAE"),
            "rmse": g("RMSE"), "pct3": g("% à ±3 médicaments"), "r2_train": g("R² entraînement")}


@bp.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            result = predict(form)
        except (ValueError, RuntimeError) as e:
            error = str(e)
    return render_template("medicaments.html", num_fields=NUM_FIELDS, cat_fields=CAT_FIELDS, form=form,
                           result=result, error=error, load_error=LOAD_ERROR, m=metrics())


@bp.get("/api/schema")
def api_schema():
    if LOAD_ERROR:
        return jsonify(erreur=LOAD_ERROR), 503
    return jsonify(numeriques=[{k: v for k, v in f.items() if k != "options"} | ({"options": [o[0] for o in f["options"]]} if f["kind"] == "age" else {}) for f in NUM_FIELDS],
                   categorielles=[{"name": f["name"], "valeurs": [o[0] for o in f["options"]]} for f in CAT_FIELDS],
                   metriques=metrics())


@bp.route("/api/predict", methods=["POST"])
def api_predict():
    if LOAD_ERROR:
        return jsonify(erreur=LOAD_ERROR), 503
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify(erreur="Corps JSON attendu (objet)."), 400
    try:
        return jsonify(predict(payload))
    except ValueError as e:
        return jsonify(erreur=str(e)), 422
