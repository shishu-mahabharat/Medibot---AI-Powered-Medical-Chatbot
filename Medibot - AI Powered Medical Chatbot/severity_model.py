from pathlib import Path
import os
import time
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
DATA_PATH = DATA_DIR / 'data_symptoms.csv'
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_PATH = MODEL_DIR / 'severity.pkl'

class SeverityModelCore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer='word')
        self.clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.fitted = False

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.fitted = True
        return self

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X), self.clf.classes_

    def save(self, path):
        joblib.dump({'v': self.vectorizer, 'c': self.clf}, path)

    @staticmethod
    def load(path):
        obj = joblib.load(path)
        m = SeverityModelCore()
        m.vectorizer = obj['v']
        m.clf = obj['c']
        m.fitted = True
        return m

def infer_bucket(symptoms):
    s = set(symptoms or [])
    if {'chest pain', 'shortness of breath', 'difficulty breathing', 'unconscious'} & s:
        return 'emergency'
    if 'fever' in s and 'cough' in s:
        return 'flu_like'
    if 'headache' in s:
        return 'migraine_like'
    return 'unknown'

def collect_training_rows():
    rows = []
    if not DATA_DIR.exists():
        return pd.DataFrame(columns=['symptoms', 'label'])
    for fname in os.listdir(DATA_DIR):
        fpath = DATA_DIR / fname
        if not fpath.is_file():
            continue
        lower = fname.lower()
        if not lower.endswith('.csv'):
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        cols = [c.strip() for c in df.columns]
        lcols = [c.lower() for c in cols]
        if 'symptoms' in lcols and 'label' in lcols:
            sidx = lcols.index('symptoms')
            lidx = lcols.index('label')
            for _, r in df.iterrows():
                s = str(r[cols[sidx]]).strip()
                l = str(r[cols[lidx]]).strip()
                if s:
                    rows.append({'symptoms': s, 'label': l})
            continue
        if 'symptom' in lcols and 'severity' in lcols:
            sidx = lcols.index('symptom')
            vidx = lcols.index('severity')
            for _, r in df.iterrows():
                s = str(r[cols[sidx]]).strip().lower()
                try:
                    v = int(r[cols[vidx]])
                except Exception:
                    v = 0
                if v >= 4:
                    label = 'emergency'
                elif v >= 2:
                    label = 'flu_like'
                elif v == 1:
                    label = 'migraine_like'
                else:
                    label = 'unknown'
                if s:
                    rows.append({'symptoms': s, 'label': label})
            continue
        if 'disease' in lcols:
            non_sym = {'disease', 'age', 'gender', 'blood pressure', 'cholesterol level', 'outcome variable'}
            symptom_cols = [cols[i] for i, k in enumerate(lcols) if k not in non_sym]
            for _, r in df.iterrows():
                syms = []
                for c in symptom_cols:
                    val = str(r[c]).strip().lower()
                    if val and val not in {'0', 'no', 'false', 'none', 'null', 'n'}:
                        syms.append(c.strip().lower())
                syms = list(dict.fromkeys(syms))
                if syms:
                    label = infer_bucket(syms)
                    rows.append({'symptoms': ' '.join(syms), 'label': label})
            continue
        symptom_like = [c for c in lcols if c.startswith('symptom_')]
        if symptom_like:
            for _, r in df.iterrows():
                syms = []
                for c in symptom_like:
                    val = str(r[cols[lcols.index(c)]]).strip().lower()
                    if val:
                        syms.append(val.replace('_', ' '))
                syms = list(dict.fromkeys(syms))
                if syms:
                    label = infer_bucket(syms)
                    rows.append({'symptoms': ' '.join(syms), 'label': label})
            continue
    if not rows:
        return pd.read_csv(DATA_PATH)
    df = pd.DataFrame(rows)
    df = df.dropna()
    df = df[df['symptoms'].astype(str).str.len() > 0]
    df = df.drop_duplicates()
    return df

def fit_save(data_path=None, model_out=str(MODEL_PATH)):
    if data_path:
        df = pd.read_csv(data_path)
    else:
        df = collect_training_rows()
    texts = df['symptoms'].astype(str).values
    labels = df['label'].astype(str).values
    core = SeverityModelCore().fit(texts, labels)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    core.save(model_out)
    return model_out

def load_artifact(model_path=str(MODEL_PATH)):
    p = Path(model_path)
    if p.exists():
        return SeverityModelCore.load(model_path)
    return None

def pull_symptoms(text):
    t = (text or '').lower()
    keys = ['fever','cough','headache','chest pain','shortness of breath','difficulty breathing','unconscious','nausea','fatigue','sore throat']
    found = []
    for k in keys:
        if k in t:
            found.append(k)
    return list(dict.fromkeys(found))

def build_actions(level, emergency):
    if emergency:
        return ['Call emergency services', 'Do not delay medical attention']
    if level == 'severe':
        return ['Seek urgent medical care', 'Avoid strenuous activity', 'Hydrate and rest']
    if level == 'moderate':
        return ['Consult a doctor soon', 'Monitor symptoms', 'Hydrate and rest']
    return ['Rest', 'Hydrate', 'Use over-the-counter relief if needed']

def predict_and_confidence(text, model=None, model_path=str(MODEL_PATH)):
    if model is None:
        model = load_artifact(model_path)
    if model is None:
        fit_save(None, model_path)
        model = load_artifact(model_path)
    probs, classes = model.predict_proba([text])
    vec = probs[0]
    idx = int(np.argmax(vec))
    raw_label = str(classes[idx])
    conf = float(vec[idx])
    symptoms = pull_symptoms(text)
    emergency_flag = ('chest pain' in symptoms) or ('shortness of breath' in symptoms) or ('difficulty breathing' in symptoms) or ('unconscious' in symptoms)
    level_map = {'emergency': 'severe', 'flu_like': 'moderate', 'migraine_like': 'moderate', 'unknown': 'mild'}
    severity = level_map.get(raw_label, 'mild')
    dist = {str(c): float(p) for c, p in zip(classes, vec)}
    sset = set(symptoms)
    reasons = []
    if 'fever' in sset and 'cough' in sset:
        reasons.append('Flu or common cold')
    if 'fever' in sset and 'headache' in sset:
        reasons.append('Viral fever')
    if 'headache' in sset:
        reasons.append('Dehydration')
    if 'sore throat' in sset:
        reasons.append('Throat infection')
    if 'chest pain' in sset or 'shortness of breath' in sset or 'difficulty breathing' in sset:
        reasons.append('Cardiorespiratory issue')
    if not reasons:
        reasons.append('Non-specific viral syndrome')
    actions = build_actions(severity, emergency_flag)
    em = []
    if severity == 'severe':
        em.extend(['Severe chest pain', 'Shortness of breath', 'Unconsciousness'])
    if 'fever' in sset:
        em.append('Fever > 39°C')
    if 'headache' in sset:
        em.append('Severe headache')
    timestamp = int(time.time() * 1000)
    return {
        "assessment": severity,
        "confidence": int(round(conf * 100)),
        "reasons": reasons,
        "recommendations": list(dict.fromkeys(actions)),
        "emergency_signs": list(dict.fromkeys(em)),
        "timestamp": timestamp,
        "symptoms": symptoms,
        "distribution": dist
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default='')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.train:
        out = fit_save()
        print(out)
    else:
        res = predict_and_confidence(args.text)
        print(json.dumps(res))
