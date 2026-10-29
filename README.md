# Prédiction de l'issue du séjour – Gradient Boosting

Application Flask autonome (projet ML « Parcours hospitalier des patients diabétiques », Eya Ouali).

## Lancer en local
    pip install -r requirements.txt
    python app.py          # http://127.0.0.1:5000

## API JSON
    curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
         -d '{"AGE": 70, "UREA": 120, "CREATININE": 3, "EF": 25, "SHOCK": 1, "EMERGENCY": 1}'

## Déployer sur Render
1. Pousser ce dossier sur un dépôt GitHub (avec `pipeline_deploiement.joblib`).
2. Render > New > Web Service > connecter le dépôt.
3. Build : `pip install -r requirements.txt` · Start : `gunicorn app:app` (ou utiliser `render.yaml`).
4. Variable d'environnement `PYTHON_VERSION` = 3.12.7 (ou une version 3.11+ disponible).

## Important
- `scikit-learn==1.8.0` est figé : le fichier .joblib doit être lu avec la version qui l'a créé.
- Le fichier `pipeline_deploiement.joblib` est généré par le notebook (section 3.9) et contient le modèle, l'imputeur, les bornes, la calibration du score de décès et les niveaux de risque.
- Les probabilités brutes du modèle sont gonflées par les poids de classes. L'application affiche donc un **risque de décès calibré** (Platt sur prédictions hors-échantillon du train). Vérification sur le test : Brier 0,0305 → 0,0249, et taux de décès prédit proche du taux observé par niveau.
- Niveaux (test) : faible < 1 % (0,3 % de décès observés), modéré 1–5 % (1,5 %), élevé 5–20 % (10,1 %), très élevé ≥ 20 % (66,7 %).
- Sur l'offre gratuite de Render, le premier chargement après inactivité peut prendre une minute.
