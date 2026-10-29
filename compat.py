"""Compatibilité scikit-learn.

Le modèle « Durée de séjour » a été entraîné avec scikit-learn 1.6.1, celui de « Issue du séjour »
avec la 1.8.0. Un site ne peut installer qu'une seule version (la 1.8.0, voir requirements.txt).
Entre ces versions, scikit-learn a renommé une classe interne que le fichier .joblib de 1.6.1
référence encore ; on la redéfinit ici pour que ce fichier puisse être relu.
À importer AVANT joblib.load(...).
"""
import sklearn.compose._column_transformer as _ct

if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        def __init__(self, lst=(), *, future_dtype=None, warning_was_emitted=False, warning_enabled=True):
            super().__init__(lst)
            self.future_dtype = future_dtype
            self.warning_was_emitted = warning_was_emitted
            self.warning_enabled = warning_enabled

    _ct._RemainderColsList = _RemainderColsList
