"""Point d'entrée : un seul site Flask qui regroupe les deux projets.

  /          page d'accueil (navigation)
  /sejour/   durée de séjour (scénarios A et B)      -> sejour_app.py
  /issue/    issue du séjour (risque de décès)        -> issue_app.py
  /medicaments/  nombre de médicaments (régression)   -> medicaments_app.py
  /readmission/  réadmission à 30 jours (classification) -> readmission_app.py
  /profils/  profils de santé (clustering CDC)        -> profils_app.py
  /health    état du service (utilisé par Render)

Render démarre ce fichier avec :  gunicorn app:app
"""
import logging

from flask import Flask, jsonify, render_template, request

import issue_app
import medicaments_app
import profils_app
import readmission_app
import sejour_app

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024   # 32 Ko : largement suffisant
app.config["JSON_AS_ASCII"] = False

app.register_blueprint(sejour_app.bp)
app.register_blueprint(issue_app.bp)
app.register_blueprint(medicaments_app.bp)
app.register_blueprint(readmission_app.bp)
app.register_blueprint(profils_app.bp)


@app.get("/")
def accueil():
    return render_template("accueil.html")


@app.get("/health")
def health():
    # Toujours 200 tant que le site tourne : un modèle défaillant n'empêche pas l'autre de fonctionner.
    sejour_ok = sejour_app.LOAD_ERROR is None
    med_ok = medicaments_app.LOAD_ERROR is None
    readm_ok = readmission_app.LOAD_ERROR is None
    prof_ok = profils_app.READY_ERROR is None and profils_app.ARTIFACT is not None
    return jsonify(
        statut="ok" if (sejour_ok and med_ok and readm_ok and prof_ok) else "partiel",
        sejour={"ok": sejour_ok, "modeles": sorted(sejour_app.PIPES), "detail": sejour_app.LOAD_ERROR},
        issue={"ok": True, "modele": "Gradient Boosting"},
        medicaments={"ok": med_ok, "modele": (medicaments_app.ART or {}).get("nom_modele"), "detail": medicaments_app.LOAD_ERROR},
        profils={"ok": prof_ok, "modele": (profils_app.ARTIFACT or {}).get("best_model_name"), "detail": profils_app.READY_ERROR},
        readmission={"ok": readm_ok, "modele": (readmission_app.BUNDLE or {}).get("model_name"), "detail": readmission_app.LOAD_ERROR},
    )


@app.errorhandler(404)
def not_found(_):
    if "/api/" in request.path:
        return jsonify(erreur="Route inconnue."), 404
    return render_template("accueil.html", erreur="Page introuvable."), 404


@app.errorhandler(405)
def bad_method(_):
    return jsonify(erreur="Méthode non autorisée."), 405


@app.errorhandler(413)
def too_large(_):
    return jsonify(erreur="Requête trop volumineuse."), 413


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
