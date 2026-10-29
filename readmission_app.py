"""Partie « Réadmission à 30 jours » : blueprint Flask monté sous /readmission.

Remplace l'ancienne interface Streamlit : même modèle, mêmes variables, mais servi par le site Flask commun.
Le fichier models/readmission/model.joblib est produit par  python train_readmission.py --data data/diabetic_data.csv
C'est un dictionnaire (preprocessor, model, selected_columns, input_columns, categories, threshold, top10_cutoff,
reference_scores, metrics, ...). Le formulaire est construit à partir de ce dictionnaire.
"""
import logging
import math
from pathlib import Path

import compat  # noqa: F401  (doit précéder joblib.load)
import joblib
import pandas as pd
from flask import Blueprint, jsonify, render_template, request

from readmit_core import add_derived
from readmit_core import predict as core_predict

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "readmission" / "model.joblib"
bp = Blueprint("readmission", __name__, url_prefix="/readmission")
log = logging.getLogger("readmission")

# --------------------------------------------------------------------------
# Champs du formulaire : (nom de colonne, libellé, min, max, valeur par défaut, aide)
# --------------------------------------------------------------------------
NUMERIC = [
    ("time_in_hospital", "Durée du séjour (jours)", 1, 14, 4, "Nombre de jours d'hospitalisation."),
    ("num_lab_procedures", "Examens de laboratoire", 0, 150, 43, "Pendant le séjour."),
    ("num_procedures", "Procédures (hors laboratoire)", 0, 10, 1, "Pendant le séjour."),
    ("num_medications", "Médicaments administrés", 0, 90, 15, "Nombre de médicaments distincts."),
    ("number_diagnoses", "Nombre de diagnostics", 1, 16, 8, "Diagnostics enregistrés."),
    ("number_outpatient", "Consultations externes", 0, 50, 0, "Dans l'année précédente."),
    ("number_emergency", "Passages aux urgences", 0, 80, 0, "Dans l'année précédente."),
    ("number_inpatient", "Hospitalisations précédentes", 0, 25, 0, "Dans l'année précédente."),
    ("n_meds_used", "Antidiabétiques utilisés", 0, 10, 1, "Nombre de médicaments antidiabétiques."),
    ("n_meds_changed", "Antidiabétiques dont la dose a changé", 0, 10, 0, "Dose augmentée ou diminuée."),
]
# (nom de colonne, libellé, valeur par défaut préférée)
CATEGORICAL = [
    ("gender", "Sexe", None), ("race", "Origine", "Unknown"), ("weight", "Poids (tranche, en livres)", "Unknown"),
    ("payer_code", "Assurance (code payeur)", "Unknown"), ("medical_specialty", "Spécialité du médecin admetteur", "Unknown"),
    ("admission_type", "Type d'admission", "Emergency"), ("admission_source", "Source d'admission", "EmergencyRoom"),
    ("discharge_group", "Destination à la sortie", "Home"),
    ("diag_1_group", "Diagnostic principal", "Circulatory"), ("diag_2_group", "Diagnostic secondaire 1", "Other"),
    ("diag_3_group", "Diagnostic secondaire 2", "Other"),
    ("A1Cresult", "Résultat HbA1c", "Not tested"), ("max_glu_serum", "Glycémie sérique maximale", "Not tested"),
    ("insulin", "Insuline", "No"),
]
BINARY = [("diabetesMed", "Un antidiabétique a été prescrit", True),
          ("change", "Traitement modifié pendant le séjour", False)]
AGE_BRACKETS = {f"[{a}-{a + 10})": a + 5 for a in range(0, 100, 10)}
AGE_DEFAULT = "[60-70)"
DERIVED = {"total_visits", "meds_per_day", "labs_per_day"}      # calculées par readmit_core.add_derived

DIAG = {"Diabetes": "Diabète", "Circulatory": "Cardio-vasculaire", "Respiratory": "Respiratoire", "Digestive": "Digestif",
        "Genitourinary": "Génito-urinaire", "Neoplasms": "Tumeurs", "Musculoskeletal": "Musculo-squelettique",
        "Injury": "Traumatisme", "Other": "Autre"}
VALUE_LABELS = {
    "gender": {"Female": "Femme", "Male": "Homme", "Unknown/Invalid": "Non renseigné"},
    "admission_type": {"Emergency": "Urgence", "Urgent": "Urgence différée", "Elective": "Programmée", "Other": "Autre"},
    "admission_source": {"EmergencyRoom": "Urgences", "Referral": "Orientation (médecin, clinique)",
                         "Transfer": "Transfert", "Other": "Autre"},
    "discharge_group": {"Home": "Domicile", "HomeHealth": "Domicile avec soins", "CareFacility": "Établissement de soins",
                        "HospitalTransfer": "Transfert vers un hôpital", "LeftAMA": "Sortie contre avis médical",
                        "Other": "Autre"},
    "diag_1_group": DIAG, "diag_2_group": DIAG, "diag_3_group": DIAG,
    "A1Cresult": {"Not tested": "Non réalisé", "Norm": "Normal", ">7": "Supérieur à 7 %", ">8": "Supérieur à 8 %"},
    "max_glu_serum": {"Not tested": "Non réalisée", "Norm": "Normale", ">200": "Supérieure à 200 mg/dL",
                      ">300": "Supérieure à 300 mg/dL"},
    "insulin": {"No": "Non prescrite", "Steady": "Dose stable", "Up": "Dose augmentée", "Down": "Dose diminuée"},
}


def pretty(value: str) -> str:
    out = []
    for i, ch in enumerate(value):
        if i and ch.isupper() and value[i - 1].islower():
            out.append(" ")
        out.append(ch)
    return "".join(out).replace("_", " ").replace("-", " - ")


def value_label(col: str, value: str) -> str:
    lab = VALUE_LABELS.get(col, {}).get(value)
    if lab:
        return lab
    if value in ("Unknown", "nan"):
        return "Non renseigné"
    return pretty(value) if col == "medical_specialty" else value


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
BUNDLE = None
LOAD_ERROR = None
NUM_FIELDS, CAT_FIELDS, BIN_FIELDS = [], [], []
REQUIRED_KEYS = ("preprocessor", "model", "selected_columns", "input_columns", "categories",
                 "threshold", "top10_cutoff", "reference_scores", "metrics", "model_name")


def load_model(path=None):
    """Charge le modèle et prépare le formulaire. Ne lève jamais : renseigne LOAD_ERROR."""
    global BUNDLE, LOAD_ERROR, NUM_FIELDS, CAT_FIELDS, BIN_FIELDS
    path = Path(path) if path else MODEL_PATH
    try:
        b = joblib.load(path)
        for k in REQUIRED_KEYS:
            if k not in b:
                raise KeyError(f"clé « {k} » absente du fichier")
        cols = set(b["input_columns"])
        num = [f for f in NUMERIC if f[0] in cols]
        cat = []
        for name, label, pref in CATEGORICAL:
            if name in cols:
                vals = [str(v) for v in b["categories"][name]]
                cat.append({"name": name, "label": label, "default": pref if pref in vals else vals[0],
                            "options": [(v, value_label(name, v)) for v in vals]})
        binf = [f for f in BINARY if f[0] in cols]
        handled = {f[0] for f in num} | {f["name"] for f in cat} | {f[0] for f in binf} | {"age_num"} | DERIVED
        missing = cols - handled
        if missing:
            raise KeyError("colonnes du modèle non gérées par le formulaire : " + ", ".join(sorted(missing)))
        BUNDLE, NUM_FIELDS, CAT_FIELDS, BIN_FIELDS, LOAD_ERROR = b, num, cat, binf, None
        log.info("Modèle réadmission chargé : %s", b["model_name"])
    except FileNotFoundError:
        BUNDLE, NUM_FIELDS, CAT_FIELDS, BIN_FIELDS = None, [], [], []
        LOAD_ERROR = ("Fichier du modèle introuvable : lancez « python train_readmission.py --data data/diabetic_data.csv » "
                      "puis copiez « model.joblib » dans le dossier « models/readmission/ » et redémarrez le site.")
    except Exception as exc:  # noqa: BLE001 - afficher n'importe quelle cause
        BUNDLE, NUM_FIELDS, CAT_FIELDS, BIN_FIELDS = None, [], [], []
        LOAD_ERROR = f"Impossible de charger le modèle : {exc}"
    if LOAD_ERROR:
        log.error(LOAD_ERROR)


load_model()


# --------------------------------------------------------------------------
# Prédiction
# --------------------------------------------------------------------------
def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "on", "yes", "oui")


def predict(values: dict, from_form: bool = False) -> dict:
    """values : formulaire ou JSON. Lève ValueError si l'entrée est invalide, RuntimeError si le modèle est absent.
    Pour un formulaire HTML, une case non cochée est absente : on la lit comme « non »."""
    if LOAD_ERROR:
        raise RuntimeError(LOAD_ERROR)
    row, errors = {}, []

    age = str(values.get("age", AGE_DEFAULT))
    if age not in AGE_BRACKETS:
        errors.append("« Tranche d'âge » : valeur inconnue.")
    else:
        row["age_num"] = AGE_BRACKETS[age]

    for name, label, lo, hi, default, _ in NUM_FIELDS:
        raw = values.get(name)
        if raw is None or str(raw).strip() == "":
            raw = default if not from_form else None
        if raw is None:
            errors.append(f"« {label} » est obligatoire.")
            continue
        try:
            x = float(str(raw).replace(",", "."))
        except ValueError:
            errors.append(f"« {label} » : entrez un nombre.")
            continue
        if not math.isfinite(x) or x < lo or x > hi:
            errors.append(f"« {label} » doit être compris entre {lo} et {hi}.")
            continue
        row[name] = x

    for f in CAT_FIELDS:
        v = str(values.get(f["name"], f["default"]))
        if v not in {o[0] for o in f["options"]}:
            errors.append(f"« {f['label']} » : valeur inconnue.")
        else:
            row[f["name"]] = v

    for name, _, default in BIN_FIELDS:
        if from_form:
            row[name] = int(name in values)
        else:
            row[name] = int(_truthy(values[name])) if name in values else int(default)

    if errors:
        raise ValueError(" ".join(errors))

    df = add_derived(pd.DataFrame([row]))
    res = core_predict(BUNDLE, df).iloc[0]
    score, pct = float(res.risk_score), float(res.percentile)
    top10, flagged = bool(res.top10_flag), bool(res.flagged)
    if top10:
        niveau = "Priorité de suivi"
    elif flagged:
        niveau = "Au-dessus du seuil d'alerte"
    else:
        niveau = "Sous le seuil d'alerte"
    return {"score": round(score, 3), "percentile": round(pct), "alerte": flagged, "top10": top10,
            "niveau": niveau, "modele": BUNDLE["model_name"]}


def metrics() -> dict:
    b = BUNDLE or {}
    m = b.get("metrics", {})
    return {"nom": b.get("model_name", "—"), "seuil": b.get("threshold"), "auc": m.get("test_roc_auc"),
            "pr_auc": m.get("test_pr_auc"), "rappel": m.get("recall"), "precision": m.get("precision"),
            "top10": m.get("top10_capture"), "variables": b.get("selected_variables", [])}


@bp.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            result = predict(form, from_form=True)
        except (ValueError, RuntimeError) as e:
            error = str(e)
    return render_template("readmission.html", num_fields=NUM_FIELDS, cat_fields=CAT_FIELDS, bin_fields=BIN_FIELDS,
                           age_options=[(k, f"{v - 5}-{v + 5} ans") for k, v in AGE_BRACKETS.items()],
                           age_default=AGE_DEFAULT, form=form, posted=request.method == "POST", result=result,
                           error=error, load_error=LOAD_ERROR, m=metrics())


@bp.get("/api/schema")
def api_schema():
    if LOAD_ERROR:
        return jsonify(erreur=LOAD_ERROR), 503
    return jsonify(
        age={"valeurs": list(AGE_BRACKETS), "defaut": AGE_DEFAULT},
        numeriques=[{"name": n, "label": l, "min": lo, "max": hi, "defaut": d} for n, l, lo, hi, d, _ in NUM_FIELDS],
        categorielles=[{"name": f["name"], "valeurs": [o[0] for o in f["options"]], "defaut": f["default"]} for f in CAT_FIELDS],
        binaires=[{"name": n, "defaut": d} for n, _, d in BIN_FIELDS], metriques=metrics())


@bp.post("/api/predict")
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
