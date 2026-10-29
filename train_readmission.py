"""Train the 30-day readmission models (same method as the notebook) and save the best one.

Usage:  python train_readmission.py --data data/diabetic_data.csv
Output: models/readmission/model.joblib  and  models/readmission/metrics.json
"""
import argparse, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score, precision_score,
                             f1_score, precision_recall_curve)
from xgboost import XGBClassifier

from readmit_core import clean_raw, engineer

RANDOM_STATE = 42
TOP_K = 20
warnings.filterwarnings('ignore')


def best_f1_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[np.argmax(f1)])


def main(path, out_dir):
    data = pd.read_csv(path)
    df_clean = clean_raw(data)
    df_fe, constant_cols = engineer(df_clean)
    print('Raw:', data.shape, '| cleaned:', df_clean.shape, '| engineered:', df_fe.shape)
    print('Early readmission rate: %.2f%%' % (100 * df_fe['readmitted_30'].mean()))

    y = df_fe['readmitted_30']
    X = df_fe.drop(columns='readmitted_30')
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    # ---- preprocessing (fit on train only)
    binary_cols = [c for c in X_train_raw.select_dtypes('number') if set(X_train_raw[c].unique()) <= {0, 1}]
    numeric_cols = [c for c in X_train_raw.select_dtypes('number') if c not in binary_cols]
    categorical_cols = X_train_raw.select_dtypes(exclude='number').columns.tolist()
    skew = X_train_raw[numeric_cols].skew()
    log_cols = skew[skew.abs() > 1].index.tolist()
    scale_only_cols = [c for c in numeric_cols if c not in log_cols]

    preprocessor = ColumnTransformer([
        ('log_scale', Pipeline([('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                                ('scale', StandardScaler())]), log_cols),
        ('scale', StandardScaler(), scale_only_cols),
        ('binary', 'passthrough', binary_cols),
        ('categorical', OneHotEncoder(drop='if_binary', min_frequency=0.01,
                                      handle_unknown='infrequent_if_exist', sparse_output=False), categorical_cols),
    ], verbose_feature_names_out=False)

    X_train_enc = pd.DataFrame(preprocessor.fit_transform(X_train_raw),
                               columns=preprocessor.get_feature_names_out(), index=X_train_raw.index)
    X_test_enc = pd.DataFrame(preprocessor.transform(X_test_raw),
                              columns=preprocessor.get_feature_names_out(), index=X_test_raw.index)

    variables_sorted = sorted(X_train_raw.columns, key=len, reverse=True)

    def source_variable(col):
        if col in X_train_raw.columns:
            return col
        for v in variables_sorted:
            if col.startswith(v + '_'):
                return v
        raise ValueError(col)
    col_to_var = {c: source_variable(c) for c in X_train_enc.columns}

    # ---- feature selection (per variable, train only)
    corr = X_train_raw[numeric_cols].corr(method='spearman')
    target_corr = X_train_raw[numeric_cols].corrwith(y_train, method='spearman').abs()
    to_drop = set()
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            if abs(corr.loc[a, b]) > 0.8 and a not in to_drop and b not in to_drop:
                to_drop.add(a if target_corr[a] < target_corr[b] else b)
    candidates = [v for v in X_train_raw.columns if v not in to_drop]
    cand_cols = [c for c in X_train_enc.columns if col_to_var[c] in candidates]

    cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    l1 = LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.03], penalty='l1', solver='liblinear',
                              cv=cv5, scoring='roc_auc', class_weight='balanced',
                              random_state=RANDOM_STATE, n_jobs=-1).fit(X_train_enc[cand_cols], y_train)
    l1_scores = pd.Series(l1.coef_[0], index=cand_cols).abs().groupby(col_to_var).max()
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight='balanced_subsample',
                                n_jobs=-1, random_state=RANDOM_STATE).fit(X_train_enc[cand_cols], y_train)
    rf_scores = pd.Series(rf.feature_importances_, index=cand_cols).groupby(col_to_var).sum()
    ranks = pd.DataFrame({'l1': l1_scores.rank(ascending=False), 'rf': rf_scores.rank(ascending=False)})
    ranks['avg'] = ranks.mean(axis=1)
    selected_variables = ranks.sort_values('avg').head(TOP_K).index.tolist()
    selected_columns = [c for c in X_train_enc.columns if col_to_var[c] in selected_variables]
    print('Selected variables:', selected_variables)

    # numpy arrays everywhere: avoids XGBoost's restriction on column names
    Xtr = X_train_enc[selected_columns].values
    Xte = X_test_enc[selected_columns].values

    # ---- models
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    searches = {
        'Logistic Regression': GridSearchCV(
            LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE),
            {'C': [0.001, 0.01, 0.1, 1, 10]}, scoring='roc_auc', cv=cv5, n_jobs=-1),
        'Decision Tree': GridSearchCV(
            DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {'max_depth': [3, 4, 6, 8], 'min_samples_leaf': [50, 100, 200, 400]},
            scoring='roc_auc', cv=cv5, n_jobs=-1),
        'XGBoost': RandomizedSearchCV(
            XGBClassifier(scale_pos_weight=spw, eval_metric='logloss', n_jobs=1, random_state=RANDOM_STATE),
            {'n_estimators': [100, 200, 300], 'max_depth': [2, 3, 4, 5], 'learning_rate': [0.01, 0.03, 0.05, 0.1],
             'subsample': [0.7, 0.85, 1.0], 'colsample_bytree': [0.5, 0.7, 1.0], 'min_child_weight': [1, 5, 10]},
            n_iter=20, scoring='roc_auc', cv=cv5, n_jobs=-1, random_state=RANDOM_STATE),
    }

    results, fitted = {}, {}
    for name, search in searches.items():
        search.fit(Xtr, y_train)
        model = search.best_estimator_
        oof = cross_val_predict(model, Xtr, y_train, cv=cv5, method='predict_proba', n_jobs=-1)[:, 1]
        thr = best_f1_threshold(y_train, oof)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= thr).astype(int)
        top = y_test.iloc[np.argsort(proba)[::-1][:int(0.10 * len(y_test))]]
        results[name] = {
            'cv_roc_auc_train': float(search.best_score_),
            'test_roc_auc': float(roc_auc_score(y_test, proba)),
            'test_pr_auc': float(average_precision_score(y_test, proba)),
            'recall': float(recall_score(y_test, pred)),
            'precision': float(precision_score(y_test, pred, zero_division=0)),
            'f1': float(f1_score(y_test, pred)),
            'threshold': thr,
            'top10_capture': float(top.sum() / y_test.sum()),
            'best_params': {k: (v.item() if hasattr(v, 'item') else v) for k, v in search.best_params_.items()},
        }
        fitted[name] = (model, thr, proba)
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in results[name].items()}))

    best = max(results, key=lambda k: results[k]['cv_roc_auc_train'])   # same rule as the notebook
    model, thr, proba = fitted[best]
    print('\nBest model by train CV ROC-AUC:', best)

    ref_scores = np.sort(proba)
    bundle = {
        'model_name': best,
        'preprocessor': preprocessor,
        'selected_columns': selected_columns,
        'selected_variables': selected_variables,
        'model': model,
        'threshold': thr,
        'top10_cutoff': float(np.quantile(ref_scores, 0.90)),
        'reference_scores': ref_scores,          # test-set scores, used to show a percentile in the app
        'input_columns': X_train_raw.columns.tolist(),
        'constant_cols': constant_cols,
        'categories': {c: sorted(X_train_raw[c].astype(str).unique().tolist()) for c in categorical_cols},
        'metrics': results[best],
    }
    import os
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(out_dir, 'model.joblib'), compress=3)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'best_model': best, 'all_models': results, 'selected_variables': selected_variables}, f, indent=2)
    print('Saved', os.path.join(out_dir, 'model.joblib'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/diabetic_data.csv')
    ap.add_argument('--out', default='models/readmission')
    a = ap.parse_args()
    main(a.data, a.out)
