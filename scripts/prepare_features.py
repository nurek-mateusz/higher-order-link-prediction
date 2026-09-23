import pandas as pd
from utils import load_trg


def parse_feature_type(param):
    codes = ['w', 'd', 'h', 't', 'm']
    if param == 'a':
        return codes

    if len(param) == 1:
        return [param]

    codes.remove(param[-1])

    return codes

def load_motifs(dataset, split):
    return pd.read_parquet(f'../data/motifs/{dataset}/{split}_mean.parquet')

def load_other_features(dataset, features, split):
    df = pd.read_parquet(f'../data/features/{dataset}/features_{split}.parquet')
    df = df.drop(columns=['triangle'])
    return df[features].copy()

def load_features(dataset, split, feature_type):
    feature_columns = {
        'w': ['swa', 'swg', 'swh', 'weight_min', 'weight_med', 'weight_max', 'weight_balance'],
        'd': ['sda', 'sdg', 'sdh','degree_min', 'degree_med', 'degree_max', 'degree_balance'],
        'h': ['hcn', 'hpa', 'haa', 'thcn'],
        't': ['iet_min', 'iet_med', 'iet_max', 'iet_triangle', 'burstiness_min', 'burstiness_med', 'burstiness_max', 'burstiness_triangle', 'age', 'recency'],
    }

    codes = parse_feature_type(feature_type)
    feature_names = []
    for k, v in feature_columns.items():
        if k in codes:
            feature_names += v

    feature_dfs =  []

    if feature_names:
        feature_dfs.append(load_other_features(dataset, feature_names, split).reset_index(drop=True))

    if 'm' in codes:
        feature_dfs.append(load_motifs(dataset, split).reset_index(drop=True))
        feature_names.extend(feature_dfs[-1].columns.tolist())

    df_combined = pd.concat(feature_dfs, axis=1)

    # labels = pd.read_parquet(f'../data/processed/{dataset}/y_{split}.parquet')
    _, labels = load_trg(f'../data/processed/{dataset}/trg_open_{split}.parquet', with_y=True)

    return df_combined.to_numpy(), labels, feature_names