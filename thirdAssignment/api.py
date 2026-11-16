"""REST API for the churn model.
    Before starting the server, make sure you ran python train.py.
   run with (python api.py)  http://127.0.0.1:5000

The threshold lives on the server. Because the test set probabilities are
stored, changing it and recomputing every metric is a comparison over 2800+
numbers instead of a refit, so /metrics answers instantly at any value.
Endpoints: 
    GET  /schema            the inputs, their types and allowed values
    GET  /predict?       one prediction, features as query arguments
    GET  /metrics            metrics and confusion matrix at the current threshold
    GET  /metrics/curve   the same metrics across the whole threshold range
    GET  /importance        permutation importance and model coefficients
    GET  /threshold           the current threshold and the suggested ones
    POST /threshold         change it, body {"threshold": 0.42}
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask
from flask_restful import Api, Resource
from flask_restful.reqparse import RequestParser

HERE = Path(__file__).parent

with open(HERE / 'model.pickle', 'rb') as f:
    MODEL = pickle.load(f)
with open(HERE / 'artifacts.json') as f:
    ARTIFACTS = json.load(f)

SCHEMA = ARTIFACTS['schema']
PROBABILITIES = np.array(ARTIFACTS['test_probabilities'])
TARGETS = np.array(ARTIFACTS['test_targets'])
FN_COST = ARTIFACTS['thresholds']['best_cost']['false_negative_cost']
FP_COST = ARTIFACTS['thresholds']['best_cost']['false_positive_cost']

# Mutable server state. One threshold, shared by every caller.
STATE = {'threshold': ARTIFACTS['thresholds']['best_f1']['threshold']}

app = Flask('Churn backend')
api = Api(app)


def score_at(threshold):
    """Every metric at one threshold, from the stored probabilities."""
    predictions = (PROBABILITIES >= threshold).astype(int)
    true_positive = int(((TARGETS == 1) & (predictions == 1)).sum())
    false_positive = int(((TARGETS == 0) & (predictions == 1)).sum())
    false_negative = int(((TARGETS == 1) & (predictions == 0)).sum())
    true_negative = int(((TARGETS == 0) & (predictions == 0)).sum())

    flagged = true_positive + false_positive
    actual = true_positive + false_negative
    precision = true_positive / flagged if flagged else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        'threshold': float(threshold),
        'accuracy': (true_positive + true_negative) / len(TARGETS),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion': {'tn': true_negative, 'fp': false_positive,
                      'fn': false_negative, 'tp': true_positive},
        'flagged': flagged,
        'cost': FN_COST * false_negative + FP_COST * false_positive,
    }


def cast(name, raw):
    """Turn a query string value into the type the encoder was fitted on.

    SeniorCitizen was fitted as the integers 0 and 1. Passing the string "0"
    would match no category, and handle_unknown='ignore' would encode it as
    all zeros without raising, so the prediction would be quietly wrong.
    """
    spec = SCHEMA[name]
    if spec['kind'] == 'number':
        return float(raw)
    if spec['dtype'] == 'int':
        return int(float(raw))
    return str(raw)


class Schema(Resource):
    def get(self):
        return {'schema': SCHEMA, 'model': ARTIFACTS['model']}


class Predict(Resource):
    def __init__(self):
        self.parser = RequestParser()
        for name in SCHEMA:
            self.parser.add_argument(name, required=True, location='args')

    def get(self):
        args = self.parser.parse_args()
        try:
            row = {name: cast(name, args[name]) for name in SCHEMA}
        except (TypeError, ValueError) as error:
            return {'error': f'could not read the inputs: {error}'}, 400

        unknown = {name: row[name] for name, spec in SCHEMA.items()
                   if spec['kind'] == 'category' and row[name] not in spec['values']}
        if unknown:
            return {'error': 'value not seen during training', 'fields': unknown}, 400

        probability = float(MODEL.predict_proba(pd.DataFrame([row]))[0, 1])
        threshold = STATE['threshold']
        return {
            'churn': int(probability >= threshold),
            'probability': probability,
            'threshold': threshold,
        }


class Metrics(Resource):
    def get(self):
        return score_at(STATE['threshold'])


class MetricsCurve(Resource):
    def get(self):
        points = [score_at(t) for t in np.linspace(0.0, 1.0, 201)]
        return {
            'points': [{k: p[k] for k in
                        ('threshold', 'accuracy', 'precision', 'recall', 'f1', 'cost')}
                       for p in points],
            'current': STATE['threshold'],
            'suggested': ARTIFACTS['thresholds'],
        }


class Importance(Resource):
    def get(self):
        return {
            'permutation': ARTIFACTS['importance'],
            'coefficients': ARTIFACTS['coefficients'],
            'note': ('Permutation importance is scored with roc_auc, which reads the '
                     'ranking rather than the labels, so it does not change with the '
                     'threshold. Coefficients are on the one-hot encoded columns.'),
        }


class Threshold(Resource):
    def __init__(self):
        self.parser = RequestParser()
        self.parser.add_argument('threshold', type=float, required=True, location='json')

    def get(self):
        return {'threshold': STATE['threshold'], 'suggested': ARTIFACTS['thresholds']}

    def post(self):
        value = self.parser.parse_args()['threshold']
        if not 0.0 <= value <= 1.0:
            return {'error': 'threshold must be between 0 and 1'}, 400
        STATE['threshold'] = value
        return {'threshold': value, 'metrics': score_at(value)}


api.add_resource(Schema, '/schema')
api.add_resource(Predict, '/predict')
api.add_resource(Metrics, '/metrics')
api.add_resource(MetricsCurve, '/metrics/curve')
api.add_resource(Importance, '/importance')
api.add_resource(Threshold, '/threshold')

if __name__ == '__main__':
    app.run(debug=True)
