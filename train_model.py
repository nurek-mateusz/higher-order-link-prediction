import compute_features_update as cf
import os
import sys
import numpy as np
import pandas as pd
import random
import pickle
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt


# Function to read the training and testing data from pickle files
def read_motif_features(dataset, split_type, s):
    with open('processing_dataset/' + split_type + '/' + dataset + '/' + s + '_mean.pickle', 'rb') as f:
        x = pickle.load(f)

    with open('processing_dataset/' + split_type + '/' + dataset + '/y_' + s + '.pickle', 'rb') as f:
        y = pickle.load(f)

    return x, y

def prepare_our_and_motif_features(x_train_our, x_val_our, x_test_our, y_train_our, y_val_our, y_test_our):
        x_train_our = pd.DataFrame(x_train_our)
        x_val_our = pd.DataFrame(x_val_our)
        x_test_our = pd.DataFrame(x_test_our)
        
        # Read training and testing data
        x_train_motifs, y_train_motifs = read_motif_features(dataset, split_type, 'train')
        x_val_motifs, y_val_motifs = read_motif_features(dataset, split_type, 'val')
        x_test_motifs, y_test_motifs = read_motif_features(dataset, split_type, 'test')

        if (x_train_motifs.shape[0] != x_train_our.shape[0]) or (x_val_motifs.shape[0] != x_val_our.shape[0]) or (x_test_motifs.shape[0] != x_test_our.shape[0]):
            raise ValueError(f'X shape mismatch! x_train_motifs.shape[0]={x_train_motifs.shape[0]}, x_train_our.shape[0]={x_train_our.shape[0]}, x_val_motifs.shape[0]={x_val_motifs.shape[0]}, x_val_our.shape[0]={x_val_our.shape[0]}, x_test_motifs.shape[0]={x_test_motifs.shape[0]}, x_test_our.shape[0]={x_test_our.shape[0]}')
        if (y_train_motifs != y_train_our) or (y_val_motifs != y_val_our) or (y_test_motifs != y_test_our):
            raise ValueError(f'Y shape/label order mismatch! y_train_motifs != y_train_our: {y_train_motifs != y_train_our}, y_val_motifs != y_val_our: {y_val_motifs != y_val_our}, y_test_motifs != y_test_our: {y_test_motifs != y_test_our}')
        
        x_train = pd.merge(x_train_our, x_train_motifs, left_on='triangle', right_on='index', how='inner').iloc[:,1:].to_numpy()
        x_val = pd.merge(x_val_our, x_val_motifs, left_on='triangle', right_on='index', how='inner').iloc[:,1:].to_numpy()
        x_test = pd.merge(x_test_our, x_test_motifs, left_on='triangle', right_on='index', how='inner').iloc[:,1:].to_numpy()

        return x_train, x_val, x_test, np.array(y_train_our), np.array(y_val_our), np.array(y_test_our)

def prepare_our_features(features_train, features_val, features_test, y_train, y_val, y_test):
        feature_names = ['hcn', 'degree_reinforcement', 'weight_reinforcement', 'pairwise_timescale_density', 
                         'timescale_density_balance', 'degree_balance', 'weight_balance', 'lifetime_one_edge', 'lifetime_two_edges']

        # Training set features
        x_train = []
        for i, item in enumerate(features_train):
            features = [item[feature] for feature in feature_names]
            x_train.append(features)
        x_train = np.array(x_train)

        # Validation set features
        x_val = []
        for i, item in enumerate(features_val):
            features = [item[feature] for feature in feature_names]
            x_val.append(features)
        x_val = np.array(x_val)

        # Test set features
        x_test = []
        for item in features_test:
            features = [item[feature] for feature in feature_names]
            x_test.append(features)
        x_test = np.array(x_test)

        return x_train, x_val, x_test, np.array(y_train), np.array(y_val), np.array(y_test)

def prepare_motif_features():
    # Read training and testing data
        x_train_motifs, y_train_motifs = read_motif_features(dataset, split_type, 'train')
        x_val_motifs, y_val_motifs = read_motif_features(dataset, split_type, 'val')
        x_test_motifs, y_test_motifs = read_motif_features(dataset, split_type, 'test')

        return x_train_motifs, x_val_motifs, x_test_motifs, np.array(y_train_motifs), np.array(y_val_motifs), np.array(y_test_motifs)

# Negative edge undersampling
def under_sample(x_train, y_train, ratio=1):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
    x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
    return x_resampled, y_resampled


if __name__ == "__main__":
    RANDOM_STATE = 0
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # dataset = 'email-Enron'
    # model_name = 'lr'
    n_iter = 500 # Number of random combinations to try for hyperparameters


    # ─────────────────────────────────────────────
    # PARSE ARGUMENTS
    # ─────────────────────────────────────────────

    if len(sys.argv) != 5:
        raise Exception('Wrong number of arguments')

    dataset = sys.argv[1]
    if dataset not in ['coauth-MAG-Geology', 'coauth-MAG-History', 'contact-high-school', 'contact-primary-school', 'contact-primary-school-2',
                       'email-Enron', 'email-Eu', 'NDC-classes', 'NDC-substances', 'tags-ask-ubuntu', 'threads-ask-ubuntu']:
        raise Exception('Wrong dataset name')

    # rf - Random Forest
    # xgb - XGBoost
    # dt - Decision Tree
    # lr - Logistic Regression
    # svm - Support Vector Machine
    # knn - K-Nearest Neighbors
    model_name = sys.argv[2]
    if model_name not in ['rf', 'xgb', 'dt', 'lr']:
        raise Exception('Wrong model name')
    
    # Split data based on time or number of events
    split_type = sys.argv[3]
    if split_type not in ['time', 'events']:
        raise Exception('Wrong split type')
    
    # Are motif features included?
    with_motifs = sys.argv[4]
    if with_motifs not in ['y', 'n']:
        raise Exception('Wrong with_motifs value')
    else:
        with_motifs = True if with_motifs == 'y' else False

    print(f'PARAMS: {dataset} {model_name} {split_type} {with_motifs}')

    # ─────────────────────────────────────────────
    # READ DATA
    # ─────────────────────────────────────────────

    print(f"[START] READ DATA - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # Read simplices
    cores = os.cpu_count()

    # Training set
    generator_train = cf.DataPreparation(n_workers=cores, dataset=dataset, split_type=split_type)
    generator_train.build_data_structures('train')

    # Validation set
    generator_val = cf.DataPreparation(n_workers=cores, dataset=dataset, split_type=split_type)
    generator_val.build_data_structures('val')

    # Test set
    generator_test = cf.DataPreparation(n_workers=cores, dataset=dataset, split_type=split_type)
    generator_test.build_data_structures('test')

    # Read labels
    with open(f'processing_dataset/{split_type}/{dataset}/y_train.pickle', 'rb') as file:
        y_train = pickle.load(file)

    with open(f'processing_dataset/{split_type}/{dataset}/y_val.pickle', 'rb') as file:
        y_val = pickle.load(file)

    with open(f'processing_dataset/{split_type}/{dataset}/y_test.pickle', 'rb') as file:
        y_test = pickle.load(file)

    # Get candidate open triangles
    candidates_train = generator_train.generate_candidate_triangles('train')
    candidates_val = generator_val.generate_candidate_triangles('val')
    candidates_test = generator_test.generate_candidate_triangles('test')

    print(f"[END] READ DATA - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────
    # CREATE FEATURES
    # ─────────────────────────────────────────────

    print(f"[START] CREATE FEATURES - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    
    # Compute features
    features_train = generator_train.calculate_triangle_features(candidates_train)
    features_val = generator_val.calculate_triangle_features(candidates_val)
    features_test = generator_test.calculate_triangle_features(candidates_test)

    # ─────────────────────────────────────────────
    # PREPARE FEATURES AND LABELS AS NUMPY ARRAYS
    # ─────────────────────────────────────────────

    if with_motifs:
        x_train, x_val, x_test, y_train, y_val, y_test = prepare_our_and_motif_features(features_train, features_val, features_test, y_train, y_val, y_test)
    else:
        x_train, x_val, x_test, y_train, y_val, y_test = prepare_our_features(features_train, features_val, features_test, y_train, y_val, y_test)

    print(f'{len(x_train)}, {len(y_train)}')
    print(f'{len(x_val)}, {len(y_val)}')
    print(f'{len(x_test)}, {len(y_test)}')
    
    # ─────────────────────────────────────────────
    # UNDERSAMPLING AND NORMALIZATION
    # ─────────────────────────────────────────────

    print(f'Before undersampling train set: zeros: {np.count_nonzero(y_train == 0)}, ones: {np.count_nonzero(y_train == 1)}')

    try:
        x_train, y_train = under_sample(x_train, y_train, ratio=0.33)
    except ValueError as e:
        # IF ValueError: The specified ratio required to generate new sample in the majority class while trying to remove samples. Please increase the ratio.
        # DO NOTHING - undersampling is not necessery because class are more or less equal
        print(e)

    print(f'After undersampling train set: zeros: {np.count_nonzero(y_train == 0)}, ones: {np.count_nonzero(y_train == 1)}')

    # Nromalize features
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    print(f"[END] CREATE FEATURES - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────
    # TRAIN MODEL
    # ─────────────────────────────────────────────

    print(f"[START] TRAIN MODEL - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # Hyperparameters
    if model_name == 'rf':
        model_constructor = RandomForestClassifier
        param_dist = {
            'n_estimators': [200, 400, 600],
            'max_depth': [None, 10, 20, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5, 0.7, 1.0],
            'bootstrap': [True],
            'oob_score': [True, False],
            'max_samples': [None, 0.7, 0.9, 1.0],
            'min_impurity_decrease': [0.0, 0.001, 0.01],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'ccp_alpha': [0.0, 0.001, 0.01],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'n_jobs': [-1],
            'random_state': [RANDOM_STATE],
        }
    elif model_name == 'xgb':
        model_constructor = xgb.XGBClassifier
        param_dist = {
            'n_estimators': [200, 400, 800, 1200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5, 10],
            'gamma': [0, 0.05, 0.1, 0.2, 1, 2, 3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'colsample_bylevel': [0.6, 0.8, 1.0],
            'colsample_bynode': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10],
            'reg_lambda': [0.01, 0.1, 1, 5, 10, 50],
            'scale_pos_weight': [1, 3, 6, 10, 20],  # tune around neg/pos ratio
            'n_jobs': [-1],
            'random_state': [RANDOM_STATE],
        }
    elif model_name == 'lr':
        model_constructor = LogisticRegression
        param_dist = {
            'penalty': ['l2', 'l1'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['liblinear', 'saga'],  # l1: liblinear/saga; l2: both
            'max_iter': [200, 500, 1000],
            'fit_intercept': [True, False],
            'class_weight': [None, 'balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:5}, {0:1, 1:10}],
            'random_state': [RANDOM_STATE],
        }

    best_score = 0
    best_params = None

    # Perform random search
    for i in range(n_iter):
        # Randomly sample parameters
        params = {key: random.choice(values) for key, values in param_dist.items()}
        
        # Create and train model
        model = model_constructor()
        model.set_params(**params)
        model.fit(x_train_scaled, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict_proba(x_val_scaled)[:, 1]
        avg_prec = average_precision_score(y_val, y_val_pred)
        auc_score = roc_auc_score(y_val, y_val_pred)
        performance = avg_prec / (sum(y_val) / len(y_val)) # avg_prec / random_baseline
        
        # Set which metric to use to choose the best model
        score = performance
        
        # Track best model
        if score > best_score:
            best_score = score
            best_params = params
            
        print(f"Iteration {i+1}/{n_iter}: Score = {score:.4f}, Best = {best_score:.4f}")

    print("\nBest parameters:", best_params)
    print("Best validation score:", best_score)

    # Final undersampling and data normalization
    try:
        x_val, y_val = under_sample(x_val, y_val, ratio=0.33)
    except ValueError as e:
        # IF ValueError: The specified ratio required to generate new sample in the majority class while trying to remove samples. Please increase the ratio.
        # DO NOTHING - undersampling is not necessery because class are more or less equal
        print(e)

    scaler = MinMaxScaler()
    x_val = scaler.fit_transform(x_val)
    x_test = scaler.transform(x_test)

    # Fit final model
    model = model_constructor()
    model.set_params(**best_params)
    model.fit(x_val, y_val)

    # Final evaluation on test set
    y_test_pred = model.predict_proba(x_test)[:, 1]
    avg_prec = average_precision_score(y_test, y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_pred)
    performance = avg_prec / (sum(y_test) / len(y_test)) # avg_prec / random_baseline

    print('Test set:', 'Performance', round(performance, 4), 'AP', round(avg_prec, 4), 'AUC', round(auc_score, 4))

    print(f"[END] TRAIN MODEL - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────
    # SAVE RESULTS
    # ─────────────────────────────────────────────

    print(f"[START] SAVE RESULTS - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # Prepare directiories
    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    feature_set = 'our_and_motifs' if with_motifs else 'our'
    results_base = f"results_{feature_set}/{split_type}/{dataset}"
    model_dir = f"{results_base}/best_model"
    params_dir = f"{results_base}/best_params"
    metrics_dir = f"{results_base}/metrics"

    for d in [model_dir, params_dir, metrics_dir]:
        ensure_dir(d)

    # Save metrics
    with open(f"{metrics_dir}/test_{model_name}.csv", "w") as f:
        f.write("performance,avg_prec,auc\n")
        f.write(f"{performance},{avg_prec},{auc_score}\n")

    # Save best params
    with open(f"{params_dir}/best_params_{model_name}.txt", "w") as f:
        f.write(str(best_params))

    # Save best model
    with open(f"{model_dir}/best_model_{model_name}.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[END] SAVE RESULTS - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────
    # SHAP ANALYSIS
    # ─────────────────────────────────────────────

    print(f"[START] SHAP - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    if with_motifs:
        feature_names = ['hcn', 'degree_reinforcement', 'weight_reinforcement',
        'pairwise_timescale_density', 'timescale_density_balance',
        'degree_balance', 'weight_balance', 'lifetime_one_edge', 'lifetime_two_edges', '0_swa', '0_swg',
        '0_swh', '1_swa', '1_swg', '1_swh', '2_swa', '2_swg', '2_swh', '3_swa',
        '3_swg', '3_swh', '4_swa', '4_swg', '4_swh', '5_swa', '5_swg', '5_swh',
        '6_swa', '6_swg', '6_swh', '7_swa', '7_swg', '7_swh', '8_swa', '8_swg',
        '8_swh', '9_swa', '9_swg', '9_swh', '10_swa', '10_swg', '10_swh',
        '11_swa', '11_swg', '11_swh', '12_swa', '12_swg', '12_swh', '13_swa',
        '13_swg', '13_swh', '14_swa', '14_swg', '14_swh', '15_swa', '15_swg',
        '15_swh', '16_swa', '16_swg', '16_swh', '17_swa', '17_swg', '17_swh',
        '18_swa', '18_swg', '18_swh', '19_swa', '19_swg', '19_swh', '20_swa',
        '20_swg', '20_swh', '21_swa', '21_swg', '21_swh', '22_swa', '22_swg',
        '22_swh', '23_swa', '23_swg', '23_swh', '24_swa', '24_swg', '24_swh']
    else:
        feature_names = ['hcn', 'degree_reinforcement', 'weight_reinforcement',
        'pairwise_timescale_density', 'timescale_density_balance',
        'degree_balance', 'weight_balance', 'lifetime_one_edge', 'lifetime_two_edges']
    
    def plot_shap(shap_obj, title, X_data):
        plt.figure()
        plt.title(title)
        shap.summary_plot(shap_obj, X_data, show=False)
        plt.savefig(f'{metrics_dir}/shap_{model_name}.pdf', bbox_inches='tight')
        plt.close()

    if model_name == 'xgb':
        explainer_xgb = shap.TreeExplainer(model)
        shap_values_xgb = explainer_xgb(x_test)
        shap_values_xgb.feature_names = feature_names
        plot_shap(shap_values_xgb, "XGBoost", x_test)

    if model_name == 'rf':
        explainer_rf = shap.TreeExplainer(model)
        shap_values_rf = explainer_rf(x_test)
        shap_values_rf = shap_values_rf[:, :, 1]
        shap_values_rf.feature_names = feature_names
        plot_shap(shap_values_rf, "Random Forest", x_test)

    if model_name == 'lr':
        explainer_log = shap.LinearExplainer(model, x_val)
        shap_values_log = explainer_log(x_test)
        shap_values_log.feature_names = feature_names
        plot_shap(shap_values_log, "Logistic Regression", x_test)

        print(f"[END] SHAP - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")