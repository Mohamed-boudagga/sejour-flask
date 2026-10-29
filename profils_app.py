"""Partie « Profils de santé » (clustering CDC Diabetes, K-Means) : blueprint Flask monté sous /profils.

Charge uniquement le modèle retenu par le notebook ; il ne réentraîne rien. Ce n'est pas un outil de diagnostic.
Fichiers : models/profils/model_cdc_diabetes.joblib et models/profils/cluster_dashboard_data.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, url_for
from sklearn.neighbors import NearestNeighbors
from cluster_service import load_artifact, predict_cluster as service_predict_cluster

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = Path(os.environ.get("MODEL_ARTIFACT_PATH", BASE_DIR / "models" / "profils" / "model_cdc_diabetes.joblib"))
DASHBOARD_PATH = Path(os.environ.get("DASHBOARD_DATA_PATH", BASE_DIR / "models" / "profils" / "cluster_dashboard_data.json"))

bp = Blueprint("profils", __name__, url_prefix="/profils")

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

YES_NO = [{"value": "0", "label": "Non"}, {"value": "1", "label": "Oui"}]

def option_list(pairs):
    return [{"value": str(v), "label": label} for v, label in pairs]

FORM_GROUPS = [
    {
        "title": "Santé et antécédents",
        "description": "Indicateurs de santé déclarés dans le questionnaire.",
        "fields": [
            {"key": "HighBP", "label": "Hypertension", "kind": "select", "options": YES_NO, "help": "1 = hypertension déclarée."},
            {"key": "HighChol", "label": "Cholestérol élevé", "kind": "select", "options": YES_NO, "help": "1 = cholestérol élevé déclaré."},
            {"key": "CholCheck", "label": "Contrôle du cholestérol", "kind": "select", "options": YES_NO, "help": "1 = contrôle effectué selon le codage du jeu."},
            {"key": "Stroke", "label": "Antécédent d’AVC", "kind": "select", "options": YES_NO, "help": "Indicateur binaire du jeu de données."},
            {"key": "HeartDiseaseorAttack", "label": "Maladie cardiaque ou infarctus", "kind": "select", "options": YES_NO, "help": "Indicateur binaire du jeu de données."},
            {"key": "GenHlth", "label": "Santé générale", "kind": "select", "options": option_list([(1, "1 — Excellente"), (2, "2 — Très bonne"), (3, "3 — Bonne"), (4, "4 — Passable"), (5, "5 — Mauvaise")]), "help": "Code ordinal : 1 = excellente, 5 = mauvaise."},
            {"key": "DiffWalk", "label": "Difficulté à marcher ou monter des escaliers", "kind": "select", "options": YES_NO, "help": "1 = difficulté déclarée."},
        ],
    },
    {
        "title": "Habitudes de vie",
        "description": "Caractéristiques de mode de vie utilisées par le modèle.",
        "fields": [
            {"key": "BMI", "label": "IMC (BMI)", "kind": "number", "min": 12, "max": 98, "step": 1, "placeholder": "Ex. 27", "help": "Valeur numérique d’IMC présente dans le jeu de données."},
            {"key": "Smoker", "label": "Tabagisme", "kind": "select", "options": YES_NO, "help": "Indicateur binaire selon le codage du jeu."},
            {"key": "PhysActivity", "label": "Activité physique", "kind": "select", "options": YES_NO, "help": "1 = activité physique déclarée."},
            {"key": "Fruits", "label": "Consommation de fruits", "kind": "select", "options": YES_NO, "help": "Indicateur binaire du jeu de données."},
            {"key": "Veggies", "label": "Consommation de légumes", "kind": "select", "options": YES_NO, "help": "Indicateur binaire du jeu de données."},
            {"key": "HvyAlcoholConsump", "label": "Consommation élevée d’alcool", "kind": "select", "options": YES_NO, "help": "Indicateur binaire du jeu de données."},
        ],
    },
    {
        "title": "Accès aux soins et bien-être",
        "description": "Variables de recours aux soins et de santé physique ou mentale.",
        "fields": [
            {"key": "AnyHealthcare", "label": "Couverture ou accès aux soins", "kind": "select", "options": YES_NO, "help": "1 = couverture de santé déclarée."},
            {"key": "NoDocbcCost", "label": "Renoncement aux soins pour raisons financières", "kind": "select", "options": YES_NO, "help": "1 = impossibilité de consulter à cause du coût."},
            {"key": "MentHlth", "label": "Jours de mauvaise santé mentale", "kind": "number", "min": 0, "max": 30, "step": 1, "placeholder": "0 à 30", "help": "Nombre de jours sur les 30 derniers jours."},
            {"key": "PhysHlth", "label": "Jours de mauvaise santé physique", "kind": "number", "min": 0, "max": 30, "step": 1, "placeholder": "0 à 30", "help": "Nombre de jours sur les 30 derniers jours."},
        ],
    },
    {
        "title": "Caractéristiques sociodémographiques",
        "description": "Codes catégoriels tels qu’ils sont utilisés lors de l’entraînement.",
        "fields": [
            {"key": "Sex", "label": "Sexe (code du jeu)", "kind": "select", "options": option_list([(0, "0 — Femme"), (1, "1 — Homme")]), "help": "Valeur encodée du jeu de données ; ce champ n’est pas une appréciation de genre."},
            {"key": "Age", "label": "Classe d’âge", "kind": "select", "options": option_list([(1, "18–24 ans"), (2, "25–29 ans"), (3, "30–34 ans"), (4, "35–39 ans"), (5, "40–44 ans"), (6, "45–49 ans"), (7, "50–54 ans"), (8, "55–59 ans"), (9, "60–64 ans"), (10, "65–69 ans"), (11, "70–74 ans"), (12, "75–79 ans"), (13, "80 ans et plus")]), "help": "Code ordinal Age : 1 à 13."},
            {"key": "Education", "label": "Niveau d’éducation (code)", "kind": "select", "options": option_list([(1, "1 — Aucun / maternelle"), (2, "2 — Primaire"), (3, "3 — Secondaire incomplet"), (4, "4 — Diplôme secondaire ou équivalent"), (5, "5 — Études supérieures, 1 à 3 ans"), (6, "6 — Études supérieures, 4 ans ou plus")]), "help": "Code ordinal du jeu de données."},
            {"key": "Income", "label": "Catégorie de revenu (code)", "kind": "select", "options": option_list([(1, "1 — Moins de 10 000 $"), (2, "2 — 10 000 à moins de 15 000 $"), (3, "3 — 15 000 à moins de 20 000 $"), (4, "4 — 20 000 à moins de 25 000 $"), (5, "5 — 25 000 à moins de 35 000 $"), (6, "6 — 35 000 à moins de 50 000 $"), (7, "7 — 50 000 à moins de 75 000 $"), (8, "8 — 75 000 $ ou plus")]), "help": "Code ordinal, pas un montant monétaire exact."},
        ],
    },
]

FEATURE_META = {
    "HighBP": {"label": "Hypertension", "type": "binary"},
    "HighChol": {"label": "Cholestérol élevé", "type": "binary"},
    "CholCheck": {"label": "Contrôle du cholestérol", "type": "binary"},
    "BMI": {"label": "IMC moyen", "type": "numeric"},
    "Smoker": {"label": "Tabagisme", "type": "binary"},
    "Stroke": {"label": "Antécédent d’AVC", "type": "binary"},
    "HeartDiseaseorAttack": {"label": "Maladie cardiaque ou infarctus", "type": "binary"},
    "PhysActivity": {"label": "Activité physique", "type": "binary"},
    "Fruits": {"label": "Consommation de fruits", "type": "binary"},
    "Veggies": {"label": "Consommation de légumes", "type": "binary"},
    "HvyAlcoholConsump": {"label": "Consommation élevée d’alcool", "type": "binary"},
    "AnyHealthcare": {"label": "Accès aux soins", "type": "binary"},
    "NoDocbcCost": {"label": "Renoncement aux soins pour coût", "type": "binary"},
    "GenHlth": {"label": "Code de santé générale", "type": "ordinal"},
    "MentHlth": {"label": "Jours de mauvaise santé mentale", "type": "numeric"},
    "PhysHlth": {"label": "Jours de mauvaise santé physique", "type": "numeric"},
    "DiffWalk": {"label": "Difficulté à marcher", "type": "binary"},
    "Sex": {"label": "Code du sexe", "type": "ordinal"},
    "Age": {"label": "Classe d’âge", "type": "ordinal"},
    "Education": {"label": "Code d’éducation", "type": "ordinal"},
    "Income": {"label": "Catégorie de revenu", "type": "ordinal"},
}

# Data is generated by the notebook on actual UCI data. Missing files are handled
# gracefully so the deployment project can be opened before export has been run.
ARTIFACT: dict[str, Any] | None = None
DASHBOARD: dict[str, Any] | None = None
NN_MODEL: NearestNeighbors | None = None
READY_ERROR: str | None = None


def _safe_load() -> None:
    global ARTIFACT, DASHBOARD, NN_MODEL, READY_ERROR
    errors = []
    try:
        if not ARTIFACT_PATH.is_file():
            errors.append(f"Fichier du modèle absent : {ARTIFACT_PATH.name}")
        else:
            candidate = load_artifact(ARTIFACT_PATH)
            required = {"scaler", "model", "best_model_name", "selected_features", "fill_values"}
            absent = sorted(required.difference(candidate))
            if absent:
                errors.append("Artefact incomplet : " + ", ".join(absent))
            elif list(candidate["selected_features"]) != FEATURES:
                errors.append("La liste ordonnée des variables de l’artefact ne correspond pas au formulaire de l’application.")
            else:
                ARTIFACT = candidate
                if candidate["best_model_name"] == "DBSCAN":
                    refs = np.asarray(candidate.get("dbscan_reference_points", []), dtype=float)
                    labels = np.asarray(candidate.get("dbscan_reference_labels", []), dtype=int)
                    if refs.size == 0:
                        NN_MODEL = None
                    elif refs.ndim == 2 and len(refs) == len(labels) and not np.any(labels == -1):
                        NN_MODEL = NearestNeighbors(n_neighbors=1, metric=candidate.get("dbscan_metric", "euclidean"))
                        NN_MODEL.fit(refs)
                    else:
                        ARTIFACT = None
                        errors.append("Référentiel DBSCAN absent ou incohérent dans l’artefact.")
        if not DASHBOARD_PATH.is_file():
            errors.append(f"Fichier descriptif absent : {DASHBOARD_PATH.name}")
        else:
            data = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
            if data.get("selected_model") and ARTIFACT and data["selected_model"] != ARTIFACT["best_model_name"]:
                errors.append("Le modèle indiqué dans les statistiques ne correspond pas au modèle sauvegardé.")
            else:
                DASHBOARD = data
    except Exception as exc:  # Provide an explicit diagnostic on the setup page.
        ARTIFACT = None
        DASHBOARD = None
        NN_MODEL = None
        errors.append(f"Erreur de chargement : {type(exc).__name__}: {exc}")
    READY_ERROR = " · ".join(errors) if errors else None


_safe_load()


def _cluster_profile(cluster_id: int) -> dict[str, Any] | None:
    if not DASHBOARD:
        return None
    for item in DASHBOARD.get("cluster_profiles", []):
        if int(item["cluster_id"]) == int(cluster_id):
            return item
    return None


def _cluster_size(cluster_id: int) -> dict[str, Any] | None:
    if not DASHBOARD:
        return None
    for item in DASHBOARD.get("cluster_sizes", []):
        if int(item["cluster_id"]) == int(cluster_id):
            return item
    return None


def _make_highlights(profile: dict[str, Any] | None) -> list[dict[str, str]]:
    if not profile or not DASHBOARD:
        return []
    means = profile.get("values", {})
    overall = DASHBOARD.get("overall_means", {})
    highlights = []
    for key in ["BMI", "HighBP", "HighChol", "PhysActivity", "GenHlth"]:
        value = means.get(key)
        base = overall.get(key)
        if value is None:
            continue
        label = FEATURE_META.get(key, {}).get("label", key)
        if key in {"HighBP", "HighChol", "PhysActivity"}:
            value_text = f"{100 * float(value):.1f} %"
            base_text = f"{100 * float(base):.1f} %" if base is not None else "—"
            note = "Proportion observée dans le groupe"
            if key == "HighBP": label = "Hypertension déclarée"
            elif key == "HighChol": label = "Cholestérol élevé déclaré"
            elif key == "PhysActivity": label = "Activité physique déclarée"
        elif key == "BMI":
            value_text = f"{float(value):.1f}"
            base_text = f"{float(base):.1f}" if base is not None else "—"
            note = "Moyenne de l’IMC dans le groupe"
        else:
            value_text = f"{float(value):.2f} / 5"
            base_text = f"{float(base):.2f} / 5" if base is not None else "—"
            note = "Moyenne du code ordinal (1 = excellente, 5 = mauvaise)"
        highlights.append({"label": label, "value": value_text, "overall": base_text, "note": note})
    return highlights


def _format_metric(value, digits=3):
    try:
        number = float(value)
        return f"{number:.{digits}f}" if np.isfinite(number) else "N/D"
    except (TypeError, ValueError):
        return "N/D"


@bp.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None
    form_values = {}
    if request.method == "POST":
        form_values = request.form.to_dict()
        if ARTIFACT is None:
            error_message = "Le modèle n’est pas encore disponible. Suivez les étapes de préparation indiquées dans la zone de configuration."
        else:
            try:
                row = {}
                for key in FEATURES:
                    raw = request.form.get(key, "").strip()
                    if raw == "":
                        raise ValueError(f"Le champ « {key} » est obligatoire.")
                    value = float(raw)
                    if not np.isfinite(value):
                        raise ValueError(f"La valeur de « {key} » n’est pas valide.")
                    row[key] = value
                # Enforce the questionnaire codes expected by the trained feature set.
                for key in ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"]:
                    if row[key] not in (0, 1):
                        raise ValueError(f"Le champ « {key} » doit valoir 0 ou 1.")
                for key, low, high in [("BMI", 12, 98), ("GenHlth", 1, 5), ("MentHlth", 0, 30), ("PhysHlth", 0, 30), ("Age", 1, 13), ("Education", 1, 6), ("Income", 1, 8)]:
                    if row[key] < low or row[key] > high:
                        raise ValueError(f"Le champ « {key} » doit être compris entre {low} et {high}.")
                    if key != "BMI" and not row[key].is_integer():
                        raise ValueError(f"Le champ « {key} » doit être un code entier.")
                cluster_id, info = service_predict_cluster(row, ARTIFACT, nearest_model=NN_MODEL)
                profile = _cluster_profile(cluster_id)
                size_info = _cluster_size(cluster_id) or {}
                if cluster_id == -1 and ARTIFACT["best_model_name"] == "DBSCAN":
                    cluster_title = "Profil non affecté (bruit DBSCAN)"
                    explanatory_text = "Le profil est trop éloigné des observations de référence non bruitées selon le seuil eps retenu. Il n’est pas forcé dans un groupe."
                else:
                    cluster_title = f"Cluster {cluster_id}"
                    explanatory_text = "L’affectation décrit la proximité statistique avec un groupe de l’échantillon. Elle ne constitue ni une probabilité ni un diagnostic médical."
                prediction = {
                    "cluster_id": cluster_id,
                    "cluster_title": cluster_title,
                    "model_name": ARTIFACT["best_model_name"],
                    "explanation": explanatory_text,
                    "highlights": _make_highlights(profile),
                    "distance": info.get("distance"),
                    "distance_label": info.get("distance_label"),
                    "distance_threshold": info.get("distance_threshold"),
                    "profile_found": profile is not None,
                    "cluster_count": size_info.get("count"),
                    "cluster_percentage": size_info.get("percentage"),
                    "diabetes_prevalence_pct": (profile or {}).get("diabetes_prevalence_pct"),
                    "target_distribution_pct": (profile or {}).get("target_distribution_pct", {}),
                }
            except (ValueError, TypeError, KeyError) as exc:
                error_message = str(exc)

    ready = ARTIFACT is not None and DASHBOARD is not None and READY_ERROR is None
    summary = DASHBOARD.get("summary", {"sample_size": DASHBOARD.get("sample_size"), "source_rows": DASHBOARD.get("source_rows"), "source_columns": DASHBOARD.get("source_columns")}) if DASHBOARD else {}
    metrics = DASHBOARD.get("metrics", {}) if DASHBOARD else {}
    return render_template(
        "profils.html",
        ready=ready,
        ready_error=READY_ERROR,
        artifact_loaded=ARTIFACT is not None,
        dashboard_loaded=DASHBOARD is not None,
        dashboard=DASHBOARD,
        summary=summary,
        metrics=metrics,
        form_groups=FORM_GROUPS,
        feature_count=len(FEATURES),
        prediction=prediction,
        error_message=error_message,
        form_values=form_values,
        format_metric=_format_metric,
        model_name=ARTIFACT.get("best_model_name") if ARTIFACT else (DASHBOARD or {}).get("selected_model", "—"),
    )


@bp.route("/api/predict", methods=["POST"])
def api_predict():
    """API JSON : reçoit un dictionnaire de variables et retourne le groupe descriptif."""
    if ARTIFACT is None or DASHBOARD is None or READY_ERROR:
        return {"error": "Application non prête", "details": READY_ERROR}, 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {"error": "Le corps doit être un objet JSON de type dictionnaire."}, 400
    try:
        cluster_id, info = service_predict_cluster(payload, ARTIFACT, nearest_model=NN_MODEL)
        profile = _cluster_profile(cluster_id) or {}
        size_info = _cluster_size(cluster_id) or {}
        if cluster_id == -1 and ARTIFACT["best_model_name"] == "DBSCAN":
            title = "Profil non affecté (bruit DBSCAN)"
        else:
            title = f"Cluster {cluster_id}"
        return {
            "cluster_id": cluster_id,
            "cluster_title": title,
            "model_name": ARTIFACT["best_model_name"],
            "description": "Affectation descriptive à partir des caractéristiques du profil ; ce résultat n’est ni un diagnostic ni une probabilité.",
            "group_size": size_info.get("count"),
            "group_percentage": size_info.get("percentage"),
            "diabetes_prevalence_pct": profile.get("diabetes_prevalence_pct"),
            "target_distribution_pct": profile.get("target_distribution_pct", {}),
            "distance": info.get("distance"),
            "distance_label": info.get("distance_label"),
            "distance_threshold": info.get("distance_threshold"),
        }
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}, 400
