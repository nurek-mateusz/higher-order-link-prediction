"""Logistic regression on all features, always evaluated on the test window.

  earlier window : standard setting - the model is trained on the tain-validation windows and
                   evaluated on the test window.
  test window    : 5-fold cross-validation within the test window - the model is trained on 4/5 of the test
                   candidates and evaluated on the remaining 1/5, five times, so every candidate is scored by a
                   model that did not see it. Same hyperparameters as the standard setting (no new tuning).
                   A diagnostic, not a forecast: training and evaluation data come from the same time window.
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import prepare_features as pf

# group A: learning beats the best baseline, group B: it does not
DATASETS = {'coauth-MAG-Geology': 'A', 'email-Enron': 'A', 'email-Eu': 'A', 'tags-ask-ubuntu': 'A',
            'threads-ask-ubuntu': 'A', 'coauth-MAG-History': 'B', 'contact-high-school': 'B',
            'contact-primary-school': 'B', 'NDC-classes': 'B', 'NDC-substances': 'B'}
N_FOLDS = 5
OUT_DIR = '../results/test_window_cv'


def read_params(row):
    params = {}
    for key, value in row.items():
        if key == 'seed':
            continue
        if isinstance(value, float) and np.isnan(value):
            value = None                                   # empty class_weight in the csv means None
        elif key == 'class_weight' and isinstance(value, str) and value.startswith('{'):
            value = ast.literal_eval(value)                # "{0: 1, 1: 3}" -> {0: 1, 1: 3}
        elif isinstance(value, np.generic):
            value = value.item()                           # numpy scalar -> plain python
        params[key] = value
    return params


if __name__ == '__main__':
    datasets = [sys.argv[1]] if len(sys.argv) == 2 else list(DATASETS)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for dataset in datasets:
        best_params = pd.read_csv(f'../results/{dataset}/best_params/lr_a.csv')
        earlier = pd.read_csv(f'../results/{dataset}/metrics/lr_a.csv').set_index('seed')['performance']

        x, y, feature_names = pf.load_features(dataset, 'test', 'a')
        y = np.asarray(y)
        swg = x[:, feature_names.index('swg')]
        swg_performance = average_precision_score(y, swg) / np.mean(y)

        for row in best_params.to_dict('records'):
            seed = int(row['seed'])
            params = read_params(row)
            p = np.zeros(len(y))

            folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fit_idx, score_idx in folds.split(x, y):
                scaler = StandardScaler()
                x_fit = scaler.fit_transform(x[fit_idx])
                x_score = scaler.transform(x[score_idx])
                y_fit = y[fit_idx]
                try:
                    rus = RandomUnderSampler(sampling_strategy=0.33, random_state=seed)
                    x_fit, y_fit = rus.fit_resample(x_fit, y_fit)
                except ValueError as e:
                    print(e)                               # classes already balanced enough, as in train_model.py

                model = LogisticRegression()
                model.set_params(**params)
                model.fit(x_fit, y_fit)
                p[score_idx] = model.predict_proba(x_score)[:, 1]

            test_window = average_precision_score(y, p) / np.mean(y)
            print(f'{dataset} seed {seed}: earlier window {earlier.loc[seed]:.4f}  '
                  f'test window {test_window:.4f}  SWG {swg_performance:.4f}')
            rows.append({'group': DATASETS.get(dataset, '?'), 'dataset': dataset, 'seed': seed,
                         'SWG': swg_performance, 'earlier_window': earlier.loc[seed], 'test_window': test_window})

    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(['group', 'dataset']).agg(
        SWG=('SWG', 'first'),
        earlier_window_mean=('earlier_window', 'mean'), earlier_window_std=('earlier_window', 'std'),
        test_window_mean=('test_window', 'mean'), test_window_std=('test_window', 'std'))

    suffix = '' if len(datasets) == len(DATASETS) else f'_{datasets[0]}'
    per_seed.to_csv(f'{OUT_DIR}/per_seed{suffix}.csv', index=False)
    summary.to_csv(f'{OUT_DIR}/summary{suffix}.csv')

    pd.set_option('display.width', 200)
    print('\n===== all features, evaluated on the test window =====')
    print(summary.round(2).to_string())