"""Shared code: cleaning + feature engineering (ported from the notebook, sections 4.1 and 4.2)
and the prediction helper used by the app."""
import numpy as np
import pandas as pd

EXPIRED_CODES = [11, 19, 20, 21]
HOSPICE_CODES = [13, 14]

MED_COLS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
            'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
            'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

UNKNOWN_CODES = {
    'admission_type_id': [5, 6, 8],
    'discharge_disposition_id': [18, 25, 26],
    'admission_source_id': [9, 15, 17, 20, 21],
}

ADMISSION_TYPE_MAP = {'1': 'Emergency', '2': 'Urgent', '3': 'Elective', '4': 'Other', '7': 'Other',
                      'Unknown': 'Unknown'}
DISCHARGE_MAP = {'1': 'Home',
                 '6': 'HomeHealth', '8': 'HomeHealth',
                 '3': 'CareFacility', '4': 'CareFacility', '5': 'CareFacility',
                 '22': 'CareFacility', '23': 'CareFacility', '24': 'CareFacility',
                 '2': 'HospitalTransfer', '9': 'HospitalTransfer', '10': 'HospitalTransfer',
                 '15': 'HospitalTransfer', '16': 'HospitalTransfer', '17': 'HospitalTransfer',
                 '27': 'HospitalTransfer', '28': 'HospitalTransfer', '29': 'HospitalTransfer',
                 '7': 'LeftAMA', '12': 'Other', '30': 'Other', 'Unknown': 'Unknown'}
ADMISSION_SOURCE_MAP = {'7': 'EmergencyRoom',
                        '1': 'Referral', '2': 'Referral', '3': 'Referral',
                        '4': 'Transfer', '5': 'Transfer', '6': 'Transfer', '10': 'Transfer',
                        '22': 'Transfer', '25': 'Transfer', '26': 'Transfer',
                        '8': 'Other', '11': 'Other', '13': 'Other', '14': 'Other', 'Unknown': 'Unknown'}


def icd9_group(code):
    """Map an ICD-9 code to a clinical family."""
    code = str(code)
    if code in ('Unknown', 'nan'):
        return 'Unknown'
    if code[0] in 'VE':
        return 'Other'
    c = float(code)
    if 250 <= c < 251:                       return 'Diabetes'
    if 390 <= c <= 459 or c == 785:          return 'Circulatory'
    if 460 <= c <= 519 or c == 786:          return 'Respiratory'
    if 520 <= c <= 579 or c == 787:          return 'Digestive'
    if 580 <= c <= 629 or c == 788:          return 'Genitourinary'
    if 140 <= c <= 239:                      return 'Neoplasms'
    if 710 <= c <= 739:                      return 'Musculoskeletal'
    if 800 <= c <= 999:                      return 'Injury'
    return 'Other'


def clean_raw(data: pd.DataFrame) -> pd.DataFrame:
    """Notebook section 4.1: raw diabetic_data.csv -> cleaned frame with target `readmitted_30`."""
    df = data.replace('?', np.nan).copy()

    for col in ['max_glu_serum', 'A1Cresult']:
        df[col] = df[col].replace('None', np.nan).fillna('Not tested')

    df = df[~df['discharge_disposition_id'].isin(EXPIRED_CODES + HOSPICE_CODES)]

    df = df.sort_values('encounter_id')
    df = df[~df.duplicated(subset='patient_nbr', keep='first')].reset_index(drop=True)

    fill_cols = ['weight', 'medical_specialty', 'payer_code', 'race', 'diag_1', 'diag_2', 'diag_3']
    df[fill_cols] = df[fill_cols].fillna('Unknown')
    df['medical_specialty'] = df['medical_specialty'].replace('PhysicianNotFound', 'Unknown')

    for col, codes in UNKNOWN_CODES.items():
        df[col] = df[col].astype(str).replace([str(c) for c in codes], 'Unknown')

    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    return df


def engineer(df_clean: pd.DataFrame, constant_cols=None):
    """Notebook section 4.2: cleaned frame -> model-ready human-readable frame (`df_fe`).
    Returns (df_fe, constant_cols)."""
    df = df_clean.copy()
    if constant_cols is None:
        constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    df = df.drop(columns=['encounter_id', 'patient_nbr', 'readmitted'] + list(constant_cols))

    for c in ['diag_1', 'diag_2', 'diag_3']:
        df[c + '_group'] = df[c].map(icd9_group)
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

    df['age_num'] = df['age'].str.extract(r'\[(\d+)-')[0].astype(int) + 5
    df = df.drop(columns='age')

    df['total_visits'] = df[['number_outpatient', 'number_emergency', 'number_inpatient']].sum(axis=1)

    med_present = [m for m in MED_COLS if m in df.columns]
    df['n_meds_used'] = (df[med_present] != 'No').sum(axis=1)
    df['n_meds_changed'] = df[med_present].isin(['Up', 'Down']).sum(axis=1)
    df = df.drop(columns=[m for m in med_present if m != 'insulin'])

    for col, mapping, new in [('admission_type_id', ADMISSION_TYPE_MAP, 'admission_type'),
                              ('discharge_disposition_id', DISCHARGE_MAP, 'discharge_group'),
                              ('admission_source_id', ADMISSION_SOURCE_MAP, 'admission_source')]:
        unmapped = set(df[col].unique()) - set(mapping)
        assert not unmapped, f'Codes without a group in {col}: {unmapped}'
        df[new] = df[col].map(mapping)
    df = df.drop(columns=list(UNKNOWN_CODES))

    df['change'] = (df['change'] == 'Ch').astype(int)
    df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

    spec = df['medical_specialty'].astype(str)
    spec = spec.mask(spec.str.startswith('Surgery'), 'Surgery').mask(spec.str.startswith('Orthopedics'), 'Orthopedics')
    df['medical_specialty'] = spec

    df['meds_per_day'] = df['num_medications'] / df['time_in_hospital']
    df['labs_per_day'] = df['num_lab_procedures'] / df['time_in_hospital']
    return df, list(constant_cols)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """For the app: compute the derived columns from user-entered fields."""
    df = df.copy()
    df['total_visits'] = df[['number_outpatient', 'number_emergency', 'number_inpatient']].sum(axis=1)
    df['meds_per_day'] = df['num_medications'] / df['time_in_hospital'].clip(lower=1)
    df['labs_per_day'] = df['num_lab_procedures'] / df['time_in_hospital'].clip(lower=1)
    return df


def predict(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """rows: DataFrame with bundle['input_columns'] (engineered, human-readable columns)."""
    X = rows[bundle['input_columns']]
    enc = bundle['preprocessor'].transform(X)
    enc = pd.DataFrame(enc, columns=bundle['preprocessor'].get_feature_names_out(), index=X.index)
    proba = bundle['model'].predict_proba(enc[bundle['selected_columns']].values)[:, 1]
    ref = bundle['reference_scores']
    out = pd.DataFrame({
        'risk_score': proba,
        'flagged': proba >= bundle['threshold'],
        'percentile': np.searchsorted(ref, proba) / len(ref) * 100,
        'top10_flag': proba >= bundle['top10_cutoff'],
    }, index=rows.index)
    return out
