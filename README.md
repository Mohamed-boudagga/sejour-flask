# Projet ML · Patients diabétiques

Un seul site Flask, deux outils :

| Adresse    | Outil                                             | Code            |
|------------|---------------------------------------------------|-----------------|
| `/`        | Accueil (navigation)                              | `app.py`        |
| `/sejour/` | Durée de séjour (scénarios A et B)                | `sejour_app.py` |
| `/issue/`  | Issue du séjour (risque de décès)                 | `issue_app.py`  |
| `/health`  | État du service                                   | `app.py`        |

API JSON : `POST /sejour/api/predict`, `GET /sejour/api/schema`, `POST /issue/api/predict`.

## Ajouter un 3e modèle plus tard
1. Créer `mon_modele_app.py` avec un `Blueprint("mon_modele", __name__, url_prefix="/mon_modele")`.
2. Mettre le modèle dans `models/mon_modele/` et la page dans `templates/mon_modele.html`.
3. Dans `app.py` : `import mon_modele_app` puis `app.register_blueprint(mon_modele_app.bp)`.
4. Ajouter un lien dans `templates/_nav.html` et une carte dans `templates/accueil.html`.

## Render
- Build : `pip install -r requirements.txt` · Start : `gunicorn app:app --workers 1 --threads 4 --timeout 120`
- Variable `PYTHON_VERSION` = `3.12.7`

## Important : versions
`requirements.txt` fige `scikit-learn==1.8.0` (version du modèle « issue »). Le modèle « durée de séjour » a été
entraîné avec la 1.6.1 : `compat.py` permet de le relire. Pour une compatibilité propre, ré-exportez ce modèle
depuis son notebook avec scikit-learn 1.8.0 et remplacez les fichiers de `models/sejour/`.
