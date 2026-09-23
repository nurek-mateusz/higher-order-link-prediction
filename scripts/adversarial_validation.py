import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import datasets


def load_motifs(dataset, split):
    return pd.read_parquet(f'../data/motifs/{dataset}/{split}_mean.parquet')

def load_features(dataset, split, features):
    df = pd.read_parquet(f'../data/features/{dataset}/features_{split}.parquet')

    feature_columns = {
        'weight': ['swa', 'swg', 'swh', 'weight_min', 'weight_med', 'weight_max', 'weight_balance'],
        'heuristics': ['hcn', 'hpa', 'haa', 'thcn'],
        'degree': ['sda', 'sdg', 'sdh','degree_min', 'degree_med', 'degree_max', 'degree_balance'],
        'time': ['iet_min', 'iet_med', 'iet_max', 'iet_triangle', 'burstiness_min', 'burstiness_med', 'burstiness_max', 'burstiness_triangle', 'age', 'recency'],
    }

    return df[feature_columns[features]].copy()

if __name__ == "__main__":
    rows = []
    for dataset in datasets:
        for split_1, split_2 in [('train', 'val'), ('val', 'test'), ('train', 'test')]:
            for features in ['motifs', 'weight', 'heuristics', 'degree', 'time']:
                if features == 'motifs':
                    df_1 = load_motifs(dataset, split_1)
                    df_2 = load_motifs(dataset, split_2)
                else:
                    df_1 = load_features(dataset, split_1, features)
                    df_2 = load_features(dataset, split_2, features)

                df_1['label'] = 0
                df_2['label'] = 1

                n_samples = min(df_1.shape[0], df_2.shape[0])
                df_1_sample = df_1.sample(n=n_samples, random_state=0)
                df_2_sample = df_2.sample(n=n_samples, random_state=0)

                df = pd.concat([df_1_sample, df_2_sample], ignore_index=True)
                x = df.drop(columns=['label']).to_numpy()
                y = df['label'].to_numpy()

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                aucs = cross_val_score(XGBClassifier(n_jobs=-1, random_state=0), x, y, cv=cv, scoring='roc_auc', n_jobs=1)

                for fold, auc in enumerate(aucs):
                    rows.append({'dataset': dataset, 'split1': split_1, 'split2': split_2, 'features': features, 'fold': fold, 'auc': float(auc)})

                print(f'{dataset}, {features}, {split_1} vs {split_2}, AUC: {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}')
                
    os.makedirs('../results', exist_ok=True)
    pd.DataFrame(rows).to_csv('../results/adversarial_validation.csv', index=False)
    

