"""Partie « Issue du séjour » (Gradient Boosting) : blueprint Flask monté sous /issue.
Issues : sortie normale / DAMA (sortie contre avis médical) / décès."""
import os

import compat  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
A = joblib.load(os.path.join(BASE, "models", "issue", "pipeline_deploiement.joblib"))
MODEL, COLS, MED = A["model"], A["columns"], A["med_cols"]
IMPUTER, BOUNDS, DEF, CUTS = A["imputer"], A["bounds"], A["defaults"], A["risk_cuts"]
CAL_A, CAL_B = A["calib"]
LEVEL_STATS = A["level_stats"]
CLASSES = ["Sortie normale", "DAMA", "Décès"]
LEVELS = A["levels"]

# Champs numériques : (nom colonne, libellé, unité, optionnel)
NUMERIC = [("AGE", "Âge", "ans", False), ("HB", "Hémoglobine", "g/dL", False), ("TLC", "Globules blancs", "×10³/µL", False),
           ("PLATELETS", "Plaquettes", "×10³/µL", False), ("GLUCOSE", "Glucose", "mg/dL", False), ("UREA", "Urée", "mg/dL", False),
           ("CREATININE", "Créatinine", "mg/dL", False), ("BNP", "BNP", "pg/mL", True), ("EF", "Fraction d'éjection (EF)", "%", True)]
HISTORY = {"SMOKING": "Tabagisme", "ALCOHOL": "Alcool", "HTN": "Hypertension", "CAD": "Maladie coronarienne",
           "PRIOR CMP": "Cardiomyopathie antérieure", "CKD": "Insuffisance rénale chronique"}
DIAG = {"RAISED CARDIAC ENZYMES": "Enzymes cardiaques élevées", "SEVERE ANAEMIA": "Anémie sévère", "ANAEMIA": "Anémie",
        "STABLE ANGINA": "Angor stable", "ACS": "Syndrome coronarien aigu", "STEMI": "Infarctus avec sus-décalage ST",
        "ATYPICAL CHEST PAIN": "Douleur thoracique atypique", "HEART FAILURE": "Insuffisance cardiaque",
        "HFREF": "IC à FE réduite", "HFNEF": "IC à FE préservée", "VALVULAR": "Valvulopathie", "CHB": "Bloc AV complet",
        "SSS": "Maladie du sinus", "AKI": "Insuffisance rénale aiguë", "CVA INFRACT": "AVC ischémique",
        "CVA BLEED": "AVC hémorragique", "AF": "Fibrillation auriculaire", "VT": "Tachycardie ventriculaire",
        "PSVT": "Tachycardie supraventriculaire", "CONGENITAL": "Cardiopathie congénitale", "UTI": "Infection urinaire",
        "NEURO CARDIOGENIC SYNCOPE": "Syncope neurocardiogénique", "ORTHOSTATIC": "Hypotension orthostatique",
        "INFECTIVE ENDOCARDITIS": "Endocardite infectieuse", "DVT": "Thrombose veineuse profonde",
        "CARDIOGENIC SHOCK": "Choc cardiogénique", "SHOCK": "Choc", "PULMONARY EMBOLISM": "Embolie pulmonaire",
        "CHEST INFECTION": "Infection thoracique"}
BINARY = list(HISTORY) + list(DIAG)


def _num(v):
    if v is None or str(v).strip() == "":
        return None
    try:
        x = float(str(v).replace(",", "."))
    except ValueError:
        raise ValueError(f"Valeur numérique invalide : « {v} »")
    if not np.isfinite(x) or x < 0:
        raise ValueError(f"Valeur hors plage : « {v} »")
    return x


def predict(form):
    """form : dict (formulaire ou JSON). Retourne un dict de résultats."""
    row = {c: DEF[c] for c in COLS}
    for c, _, _, optional in NUMERIC:
        v = _num(form.get(c))
        if c == "AGE" and v is not None and v > 120:
            raise ValueError("Âge hors plage (0–120).")
        if optional:
            row[c] = v if v is not None else np.nan
        elif v is not None:
            row[c] = v
    row["GENDER"] = 1 if str(form.get("GENDER", "M")).upper() == "M" else 0
    row["RURAL"] = 1 if str(form.get("RURAL", "U")).upper() == "R" else 0
    row["EMERGENCY"] = 1 if str(form.get("EMERGENCY", "1")) in ("1", "true", "True", "E") else 0
    for c in BINARY:
        row[c] = 1 if str(form.get(c, "")).lower() in ("1", "on", "true", "yes") else 0
    row["BNP_manquant"] = int(np.isnan(row["BNP"]))
    row["EF_manquant"] = int(np.isnan(row["EF"]))
    X = pd.DataFrame([row])[COLS]
    X[MED] = IMPUTER.transform(X[MED])              # médiane du train pour BNP/EF non renseignés
    for c, (lo, hi) in BOUNDS.items():               # mêmes bornes 1 %-99 % que l'entraînement
        X[c] = X[c].clip(lo, hi)
    p = MODEL.predict_proba(X.values)[0]             # scores bruts (pondérés par classe, non calibrés)
    q = float(np.clip(p[2], 1e-6, 1 - 1e-6))
    risk = 1 / (1 + np.exp(-(CAL_A * np.log(q / (1 - q)) + CAL_B)))   # risque de décès calibré
    k = int(np.searchsorted(CUTS, risk, side="right"))
    st = LEVEL_STATS[k]
    warn = None
    if sum(row[c] for c in BINARY) == 0:
        warn = ("Aucun antécédent ni diagnostic coché : ce profil est rare dans les données (84 patients sur 5 097), "
                "l'estimation est peu fiable. Coche les antécédents et diagnostics connus.")
    return {"avertissement": warn, "risque_deces": round(float(risk) * 100, 1), "niveau_risque": LEVELS[k],
            "taux_deces_observe_niveau": round(st["taux_observe"] * 100, 1), "patients_niveau": st["patients"],
            "scores_bruts_non_calibres": {n: round(float(v) * 100, 1) for n, v in zip(CLASSES, p)}}


bp = Blueprint("issue", __name__, url_prefix="/issue")


@bp.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            result = predict(form)
        except ValueError as e:
            error = str(e)
    return render_template("issue.html", numeric=NUMERIC, history=HISTORY, diag=DIAG, defaults=DEF,
                           form=form, result=result, error=error, levels=LEVELS)


@bp.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(predict(request.get_json(force=True, silent=True) or {}))
    except ValueError as e:
        return jsonify({"erreur": str(e)}), 400
