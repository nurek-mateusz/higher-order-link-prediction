import numpy as np
import pandas as pd
import pickle


def select_best_motifs(dataset, model_name, n_motifs, feature_type):
    our_feature_names = ['hcn', 'degree_reinforcement', 'weight_reinforcement', 'pairwise_timescale_density', 
                    'timescale_density_balance', 'degree_balance', 'weight_balance', 'lifetime_one_edge', 'lifetime_two_edges']

    if feature_type == 'b':
        directory = 'results/our_and_motifs'
    else:
        directory = 'results/motifs'

    ranking = pd.read_csv(f'../{directory}/{dataset}/metrics/shap_ranking_{model_name}_75.csv', sep=',')
    ranking = ranking[~ranking['feature_name'].isin(our_feature_names)].sort_values('rank', ascending=True)
    
    n_ranking = ranking[:n_motifs]
    return n_ranking['feature_name'].to_list()

def create_motif_features(dataset, model_name, mode, n_motifs, feature_type):
    if mode not in ['train', 'val', 'test']:
        raise Exception(f'Wrong value: {mode}. Use train, val, or test')
    
    with open('../data/motifs/' + dataset + '/' + mode + '_mean.pickle', 'rb') as file:
        motif_features = pickle.load(file)

    with open('../data/processed/' + dataset + '/y_' + mode + '.pickle', 'rb') as file:
        labels = pickle.load(file)

    if not isinstance(motif_features, pd.DataFrame):
        raise Exception('motif_features have to be DataFrame')
    
    if not isinstance(labels, list):
        raise Exception('labels have to be ndarray')
    
    # select N the best motif features, if n_motifs = 75 then no selection, use all 75 motifs
    if n_motifs < 75:
        best_motifs = select_best_motifs(dataset, model_name, n_motifs, feature_type)
        motif_features = motif_features[best_motifs]

    return motif_features, labels

def create_our_features(dataset, mode):
    if mode not in ['train', 'val', 'test']:
        raise Exception(f'Wrong value: {mode}. Use train, val, or test')
     
    # Read features
    with open(f'../data/features/{dataset}/features_{mode}.pickle', 'rb') as file:
        features = pickle.load(file)

    # Read labels
    with open(f'../data/processed/{dataset}/y_{mode}.pickle', 'rb') as file:
        labels = pickle.load(file)

    if not isinstance(features, pd.DataFrame):
        raise Exception('features have to be DataFrame')

    if not isinstance(labels, list):
        raise Exception('labels have to be ndarray')

    return features, labels

def create_our_and_motif_features(dataset, model_name, mode, n_motifs, feature_type):
    x_our, y_our = create_our_features(dataset, mode)
    x_motifs, y_motifs = create_motif_features(dataset, model_name, mode, n_motifs, feature_type)

    if x_motifs.shape[0] != x_our.shape[0]:
        raise ValueError(f'X shape mismatch! x_motifs.shape[0]={x_motifs.shape[0]}, x_our.shape[0]={x_our.shape[0]}')
    if y_motifs != y_our:
        raise ValueError(f'Y shape/label order mismatch! y_motifs != y_our: {y_motifs != y_our}')
    
    x_our_and_motifs = pd.merge(x_our, x_motifs, left_on='triangle', right_on='index', how='inner')

    if x_our_and_motifs.shape[0] != x_our.shape[0]:
        raise ValueError(f'X shape mismatch! x_our_and_motifs.shape[0]={x_our_and_motifs.shape[0]}, x_our.shape[0]={x_our.shape[0]}')
    
    if x_our_and_motifs.shape[1] != (x_our.shape[1] + x_motifs.shape[1]):
        raise ValueError(f'X shape mismatch! x_our_and_motifs.shape[1]={x_our_and_motifs.shape[1]}, (x_our.shape[1] + x_motifs.shape[1])={(x_our.shape[1] + x_motifs.shape[1])}')

    return x_our_and_motifs, y_our

def convert_to_numpy_array(df_x, y):
    if 'triangle' in df_x.columns:
        df_x = df_x.drop(columns=['triangle'])
    x_array = df_x.to_numpy()
    y_array = np.array(y)
    return x_array, y_array