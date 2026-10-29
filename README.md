# Projet ML · Patients diabétiques

Un seul site Flask, cinq outils :

| Adresse    | Outil                                             | Code            |
|------------|---------------------------------------------------|-----------------|
| `/`        | Accueil (navigation)                              | `app.py`        |
| `/sejour/` | Durée de séjour (scénarios A et B)                | `sejour_app.py` |
| `/issue/`  | Issue du séjour (risque de décès)                 | `issue_app.py`  |
| `/medicaments/` | Nombre de médicaments (régression, LightGBM) | `medicaments_app.py` |
| `/readmission/` | Réadmission à 30 jours (classification, XGBoost) | `readmission_app.py` |
| `/profils/` | Profils de santé (clustering K-Means, CDC Diabetes) | `profils_app.py` |
| `/health`  | État du service                                   | `app.py`        |

API JSON : `POST /profils/api/predict`, `POST /sejour/api/predict`, `GET /sejour/api/schema`, `POST /issue/api/predict`, `POST /medicaments/api/predict`, `GET /medicaments/api/schema`, `POST /readmission/api/predict`, `GET /readmission/api/schema`.

## Modèle « nombre de médicaments »
1. Dans le notebook, la cellule 4.6 crée `modele_medicaments.joblib`.
2. Copier ce fichier dans `models/medicaments/` (voir `LISEZ-MOI.txt`). Sans lui, la page affiche un message d'aide
   et le reste du site fonctionne.
3. Le formulaire est généré à partir des variables stockées dans le fichier : si les 7 variables changent, remplacer
   simplement le fichier.
4. Tests : `python -m unittest tests.test_medicaments -v` (modèle factice, plus un test sur le vrai fichier s'il existe).
   Le modèle est un LightGBM : `lightgbm` est dans `requirements.txt`. Si la page signale une version de
   scikit-learn différente, ré-exporter le fichier avec scikit-learn 1.8.0 (voir plus bas).

## Modèle « réadmission à 30 jours »
L'ancienne application Streamlit a été remplacée par une page Flask (`readmission_app.py`) ; le code de nettoyage et de
préparation des données est dans `readmit_core.py`, l'entraînement dans `train_readmission.py`.
1. Mettre `diabetic_data.csv` dans `data/` (ce fichier n'est pas envoyé sur GitHub).
2. Installer les mêmes versions que le site : `pip install -r requirements.txt`.
3. Lancer : `python train_readmission.py --data data/diabetic_data.csv` (quelques minutes). Cela écrit
   `models/readmission/model.joblib` et `metrics.json`.
4. Envoyer `models/readmission/model.joblib` sur GitHub. Sans lui, la page affiche un message d'aide et le reste du
   site fonctionne.
5. Tests : `python -m unittest discover -s tests -t . -v` (modèle factice sur données synthétiques).

## Ajouter un autre modèle plus tard
1. Créer `mon_modele_app.py` avec un `Blueprint("mon_modele", __name__, url_prefix="/mon_modele")`.
2. Mettre le modèle dans `models/mon_modele/` et la page dans `templates/mon_modele.html`.
3. Dans `app.py` : `import mon_modele_app` puis `app.register_blueprint(mon_modele_app.bp)`.
4. Ajouter un lien dans `templates/_nav.html` et une carte dans `templates/accueil.html`.

## Render
- Build : `pip install -r requirements.txt` · Start : `gunicorn app:app --workers 1 --threads 4 --timeout 120`
- Variable `PYTHON_VERSION` = `3.12.7`

## Profils de santé (clustering CDC)
Module `profils_app.py` + `cluster_service.py`. Fichiers : `models/profils/model_cdc_diabetes.joblib` et
`models/profils/cluster_dashboard_data.json`. Tests : `tests/test_profils.py`.

## Important : versions
Tout le site utilise `scikit-learn==1.8.0`. Les modèles « durée de séjour » (`models/sejour/`) ont été ré-exportés
avec cette version (performances identiques : MAE test 2,182 en A et 1,682 en B). Le modèle « réadmission »
a été entraîné avec ces mêmes versions. `compat.py` n'est plus nécessaire pour la durée de séjour ; il reste pour
relire d'anciens fichiers .joblib de scikit-learn 1.6.1.
Le modèle « médicaments » (`models/medicaments/modele_medicaments.joblib`, scikit-learn 1.6.1) est relu grâce à `compat.py` : ne pas supprimer ce fichier. Pour une compatibilité propre, son auteur peut le ré-exporter avec scikit-learn 1.8.0.

## Tests
`python -m unittest discover -s tests -t . -v`
