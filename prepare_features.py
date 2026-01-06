import compute_features_update as cf
import numpy as np
import pandas as pd
import pickle
from datetime import datetime


def select_best_motifs(dataset, model_name, split_type, n_motifs, feature_type):
    our_feature_names = ['hcn', 'degree_reinforcement', 'weight_reinforcement', 'pairwise_timescale_density', 
                    'timescale_density_balance', 'degree_balance', 'weight_balance', 'lifetime_one_edge', 'lifetime_two_edges']

    if feature_type == 'b':
        directory = 'results_our_and_motifs'
    else:
        directory = 'results_motifs'

    ranking = pd.read_csv(f'{directory}/{split_type}/{dataset}/metrics/shap_ranking_{model_name}.csv', sep=',')
    ranking = ranking[~ranking['feature_name'].isin(our_feature_names)].sort_values('rank', ascending=True)
    
    n_ranking = ranking[:n_motifs]
    return n_ranking['feature_name'].to_list()

def create_motif_features(dataset, model_name, split_type, mode, n_motifs, feature_type):
    if mode not in ['train', 'val', 'test']:
        raise Exception(f'Wrong value: {mode}. Use train, val, or test')
    
    with open('processing_dataset/' + split_type + '/' + dataset + '/' + mode + '_mean.pickle', 'rb') as f:
        x = pickle.load(f)

    with open('processing_dataset/' + split_type + '/' + dataset + '/y_' + mode + '.pickle', 'rb') as f:
        y = pickle.load(f)

    if not isinstance(x, pd.DataFrame):
        raise Exception('x has to be DataFrame')
    
    if not isinstance(y, list):
        raise Exception('y has to be ndarray')
    
    # select N the best motif features, if n_motifs = -1 then use all motif features
    if n_motifs > -1:
        best_motifs = select_best_motifs(dataset, model_name, split_type, n_motifs, feature_type)
        x = x[best_motifs]

    return x, y

def create_our_features(dataset, split_type, mode, n_cores):
    if mode not in ['train', 'val', 'test']:
        raise Exception(f'Wrong value: {mode}. Use train, val, or test')
     
    # Create features
    generator = cf.DataPreparation(n_workers=n_cores, dataset=dataset, split_type=split_type)
    generator.build_data_structures(mode)
    candidates = generator.generate_candidate_triangles(mode)
    features = generator.calculate_triangle_features(candidates)

    # Read labels
    with open(f'processing_dataset/{split_type}/{dataset}/y_{mode}.pickle', 'rb') as file:
        y = pickle.load(file)

    if not isinstance(y, list):
        raise Exception('y has to be ndarray')

    return pd.DataFrame(features), y

def create_our_and_motif_features(dataset, model_name, split_type, mode, n_cores, n_motifs, feature_type):
        x_our, y_our = create_our_features(dataset, split_type, mode, n_cores)
        x_motifs, y_motifs = create_motif_features(dataset, split_type, mode)

        # select N the best motif features, if n_motifs = -1 then use all motif features
        if n_motifs > -1:
            best_motifs = select_best_motifs(dataset, model_name, split_type, n_motifs, feature_type)
            x_motifs = x_motifs[best_motifs]

        if x_motifs.shape[0] != x_our.shape[0]:
            raise ValueError(f'X shape mismatch! x_motifs.shape[0]={x_motifs.shape[0]}, x_our.shape[0]={x_our.shape[0]}')
        if y_motifs != y_our:
            raise ValueError(f'Y shape/label order mismatch! y_motifs != y_our: {y_motifs != y_our}')
        
        x_our_and_motifs = pd.merge(x_our, x_motifs, left_on='triangle', right_on='index', how='inner')

        return x_our_and_motifs, y_our

def convert_to_numpy_array(df_x, y):
    if 'triangle' in df_x.columns:
        df_x = df_x.drop(columns=['triangle'])
    x_array = df_x.to_numpy()
    y_array = np.array(y)
    return x_array, y_array