"""
Trains the model the API serves and writes the artifacts it needs.

Run this once before starting the API 
    python train.py

It rebuilds the logistic regression that was among the best models in the
second assignment, using the parameters that search found, and saves:

    model.pickle  - the fitted pipeline, preprocessing included
    artifacts.json - test set probabilities, feature importance, input schema
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
DATA = HERE.parent / 'telco_churn.csv'
RANDOM_STATE = 13
NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']

# A false negative is a customer who leaves without being offered anything.
# A false positive is a discount given to someone who was staying anyway.
# Losing a customer costs more than a discount, so the two are not equal.
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1


def load_dataset():
    """Same cleanup as the second assignment."""
    dataset = pd.read_csv(DATA).drop(columns='customerID')
    # TotalCharges arrives as text because 11 rows hold a single space.
    dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
    dataset['Churn'] = (dataset['Churn'] == 'Yes').astype(int)
    return dataset


def build_pipeline(categorical):
    """The combination GridSearchCV picked: one-hot, fill missing with 0, no scaling."""
    encode = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('scale', 'passthrough'),
        ]), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ], remainder='passthrough')

    return Pipeline([
        ('encode', encode),
        ('estimator', LogisticRegression(C=0.1, class_weight='balanced',
                                         max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_schema(features, categorical):
    """Describes every input so the frontend can build a form and the API can cast.

    The dtype matters. OneHotEncoder was fitted on the integers 0 and 1 for
    SeniorCitizen, so a request carrying the string "0" would match nothing and
    handle_unknown='ignore' would quietly encode it as all zeros.
    """
    schema = {}
    for column in features.columns:
        if column in NUMERIC:
            schema[column] = {
                'kind': 'number',
                'min': float(features[column].min()),
                'max': float(features[column].max()),
                'default': float(features[column].median()),
            }
        else:
            values = sorted(features[column].dropna().unique().tolist())
            schema[column] = {
                'kind': 'category',
                'dtype': 'int' if isinstance(values[0], (int, np.integer)) else 'str',
                'values': [int(v) if isinstance(v, np.integer) else v for v in values],
                'default': features[column].mode()[0],
            }
            if schema[column]['dtype'] == 'int':
                schema[column]['default'] = int(schema[column]['default'])
    return schema


def best_threshold(targets, probabilities, score):
    """The threshold maximising `score`, searched over the probabilities themselves.

    Only the values the model actually produced can change a decision, so there
    is no point sampling a grid between them.
    """
    candidates = np.unique(probabilities)
    scores = [score(targets, (probabilities >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(scores))]), float(np.max(scores))


def total_cost(targets, predictions):
    """Negated so the same search can maximise it."""
    false_negatives = int(((targets == 1) & (predictions == 0)).sum())
    false_positives = int(((targets == 0) & (predictions == 1)).sum())
    return -(FALSE_NEGATIVE_COST * false_negatives + FALSE_POSITIVE_COST * false_positives)


def main():
    dataset = load_dataset()
    features = dataset.drop(columns='Churn')
    targets = dataset['Churn']
    categorical = [c for c in features.columns if c not in NUMERIC]

    # A 60/40 stratified split. The second assignment ended up on 75/25 after
    # feedback; this holds back more rows instead, which costs the model a
    # little and buys a tighter estimate of what it is worth.
    features_trainval, features_test, targets_trainval, targets_test = train_test_split(
        features, targets, test_size=0.4, shuffle=True,
        random_state=RANDOM_STATE, stratify=targets)

    model = build_pipeline(categorical)
    model.fit(features_trainval, targets_trainval)

    probabilities = model.predict_proba(features_test)[:, 1]

    # Importance is scored with roc_auc on purpose. It reads the ranking rather
    # than the labels, so these numbers stay valid whatever threshold the user
    # later sets. An f1 scorer would silently measure importance at 0.5 only.
    importance = permutation_importance(
        model, features_test, targets_test,
        scoring='roc_auc', n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)

    encoded_names = model.named_steps['encode'].get_feature_names_out()
    coefficients = model.named_steps['estimator'].coef_[0]

    f1_threshold, f1_value = best_threshold(targets_test, probabilities, f1_score)
    cost_threshold, cost_value = best_threshold(targets_test, probabilities, total_cost)

    artifacts = {
        'model': 'Logistic regression (C=0.1, class_weight=balanced)',
        'n_train': int(len(features_trainval)),
        'n_test': int(len(features_test)),
        'churn_rate_test': float(targets_test.mean()),
        'roc_auc': float(roc_auc_score(targets_test, probabilities)),
        'test_probabilities': [float(p) for p in probabilities],
        'test_targets': [int(t) for t in targets_test],
        'thresholds': {
            'default': 0.5,
            'best_f1': {'threshold': f1_threshold, 'f1': f1_value},
            'best_cost': {
                'threshold': cost_threshold,
                'cost': -cost_value,
                'false_negative_cost': FALSE_NEGATIVE_COST,
                'false_positive_cost': FALSE_POSITIVE_COST,
            },
        },
        'importance': [
            {'feature': name,
             'mean': float(m),
             'std': float(s)}
            for name, m, s in sorted(
                zip(features.columns, importance.importances_mean, importance.importances_std),
                key=lambda row: row[1], reverse=True)
        ],
        'coefficients': [
            {'feature': str(name), 'coefficient': float(c)}
            for name, c in sorted(zip(encoded_names, coefficients),
                                  key=lambda row: abs(row[1]), reverse=True)
        ],
        'schema': build_schema(features, categorical),
    }

    with open(HERE / 'model.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open(HERE / 'artifacts.json', 'w') as f:
        json.dump(artifacts, f, indent=1)

    print(f"trained on {artifacts['n_train']} rows, evaluated on {artifacts['n_test']}")
    print(f"roc auc            {artifacts['roc_auc']:.4f}")
    print(f"best f1 threshold  {f1_threshold:.3f}  -> f1 {f1_value:.4f}")
    print(f"best cost threshold {cost_threshold:.3f}  -> cost {-cost_value} "
          f"(fn={FALSE_NEGATIVE_COST}, fp={FALSE_POSITIVE_COST})")
    print()
    print('top features by permutation importance:')
    for row in artifacts['importance'][:5]:
        print(f"  {row['feature']:<20}{row['mean']:+.4f}")
    print('\nwrote model.pickle and artifacts.json')


if __name__ == '__main__':
    main()
