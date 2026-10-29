"""Service de prédiction du groupe retenu pour CDC Health Profile Explorer.

Le module charge un artefact créé par le notebook. Il ne réentraîne jamais le modèle.
Les clés absentes du dictionnaire sont imputées avec les valeurs mémorisées à l'entraînement ;
les clés inconnues sont refusées pour éviter une faute de saisie silencieuse.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_PATH = Path(os.environ.get("MODEL_ARTIFACT_PATH", BASE_DIR / "model_cdc_diabetes.joblib"))


def load_artifact(path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    """Charge l'artefact et valide les éléments nécessaires à une prédiction."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Artefact introuvable : {artifact_path.resolve()}")
    artifact = joblib.load(artifact_path)
    required = {"scaler", "model", "best_model_name", "selected_features", "fill_values"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError("Artefact incomplet : " + ", ".join(sorted(missing)))
    features = list(artifact["selected_features"])
    fill_values = artifact["fill_values"]
    if not features or set(features) != set(fill_values):
        raise ValueError("Liste des variables et valeurs d'imputation incohérentes.")
    if artifact["best_model_name"] not in {"K-Means", "DBSCAN"}:
        raise ValueError(f"Modèle non pris en charge : {artifact['best_model_name']!r}")
    if artifact["best_model_name"] == "DBSCAN":
        db_keys = {"dbscan_reference_points", "dbscan_reference_labels", "dbscan_eps", "dbscan_metric"}
        absent = db_keys.difference(artifact)
        if absent:
            raise ValueError("Artefact DBSCAN incomplet : " + ", ".join(sorted(absent)))
    return artifact


def predict_cluster(
    patient_dict: Mapping[str, Any],
    artifact: dict[str, Any] | None = None,
    nearest_model: NearestNeighbors | None = None,
) -> tuple[int, dict[str, Any]]:
    """Affecte une observation à un groupe avec le modèle sauvegardé.

    Les clés de variables absentes sont traitées comme des valeurs manquantes et
    imputées avec les statistiques mémorisées pendant la préparation. Une clé inconnue
    provoque une erreur explicite. La standardisation réutilise le scaler ajusté.
    Retourne (identifiant_du_cluster, informations_de_distance).
    """
    if not isinstance(patient_dict, Mapping):
        raise TypeError("patient_dict doit être un dictionnaire (clé = variable, valeur = réponse).")
    if artifact is None:
        artifact = load_artifact()

    features = list(artifact["selected_features"])
    unknown = sorted(set(patient_dict) - set(features))
    if unknown:
        raise ValueError("Variables inconnues : " + ", ".join(unknown))

    # Une clé absente équivaut à une donnée manquante : l'imputation stockée sera appliquée.
    row = {feature: patient_dict.get(feature, np.nan) for feature in features}
    frame = pd.DataFrame([row], columns=features).apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    fill_values = artifact["fill_values"]

    # Contrôles de domaine pour éviter qu'une valeur numérique absurde soit prédite silencieusement.
    binary_features = {
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
    }
    bounded_features = {
        "BMI": (12, 98), "GenHlth": (1, 5), "MentHlth": (0, 30),
        "PhysHlth": (0, 30), "Age": (1, 13), "Education": (1, 6), "Income": (1, 8),
    }
    for feature in features:
        value = frame.at[0, feature]
        if pd.isna(value):
            frame.at[0, feature] = fill_values[feature]
            continue
        if feature in binary_features and value not in (0, 1):
            raise ValueError(f"La variable {feature} doit valoir 0 ou 1.")
        if feature in bounded_features:
            lower, upper = bounded_features[feature]
            if not (lower <= value <= upper):
                raise ValueError(f"La variable {feature} doit être comprise entre {lower} et {upper}.")
            if feature != "BMI" and not float(value).is_integer():
                raise ValueError(f"La variable {feature} doit être un code entier.")

    # StandardScaler déjà ajusté : aucune ré-estimation sur le nouvel individu.
    scaled = pd.DataFrame(
        artifact["scaler"].transform(frame[features]),
        columns=features,
        index=frame.index,
    )
    model_name = artifact["best_model_name"]

    if model_name == "K-Means":
        model = artifact["model"]
        label = int(model.predict(scaled)[0])
        distances = model.transform(scaled)[0]
        return label, {
            "distance": float(np.min(distances)),
            "distance_label": "Distance au centroïde le plus proche (espace standardisé)",
            "distance_threshold": None,
            "assignment_rule": "centroïde le plus proche",
        }

    # DBSCAN : règle d'affectation approchée, pas un predict natif.
    refs = np.asarray(artifact["dbscan_reference_points"], dtype=float)
    ref_labels = np.asarray(artifact["dbscan_reference_labels"], dtype=int)
    eps = float(artifact["dbscan_eps"])
    metric = str(artifact["dbscan_metric"])
    if metric != "euclidean":
        raise ValueError("Cette règle d'affectation DBSCAN prend en charge la distance euclidienne uniquement.")
    if refs.size == 0:
        return -1, {"distance": None, "distance_label": "Distance au point non bruité le plus proche", "distance_threshold": eps, "assignment_rule": "voisin non bruité le plus proche"}
    if refs.ndim != 2 or refs.shape[1] != len(features) or len(refs) != len(ref_labels) or np.any(ref_labels == -1):
        raise ValueError("Le référentiel DBSCAN sauvegardé est incohérent.")
    nearest = nearest_model
    if nearest is None:
        nearest = NearestNeighbors(n_neighbors=1, metric=metric).fit(refs)
    distances, indices = nearest.kneighbors(scaled.to_numpy())
    distance = float(distances[0, 0])
    label = int(ref_labels[indices[0, 0]]) if distance <= eps else -1
    return label, {
        "distance": distance,
        "distance_label": "Distance au point non bruité le plus proche (espace standardisé)",
        "distance_threshold": eps,
        "assignment_rule": "voisin non bruité le plus proche avec seuil eps",
    }
