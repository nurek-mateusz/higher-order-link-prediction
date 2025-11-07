import pandas as pd
from find_motifs import *
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import average_precision_score
from imblearn.under_sampling import RandomUnderSampler
import random
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


# Function to undersample the dataset to balance positive and negative samples
def under_sample(x_train, y_train, ratio=1):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
    x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
    return x_resampled, y_resampled

# Function to read the training and testing data from pickle files
def read_data(dataset, split_type, s):
    with open('../processing_dataset/' + split_type + '/' + dataset + '/' + s + '_mean.pickle', 'rb') as f:
        x = pickle.load(f)
    # x = x.apply(lambda a: (a - a.min()) / (a.max() - a.min()))  # Normalize data

    with open('../processing_dataset/' + split_type + '/' + dataset + '/y_' + s + '.pickle', 'rb') as f:
        y = pickle.load(f)

    return x, y


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

    if len(sys.argv) != 4:
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
    if model_name not in ['rf', 'xgb', 'dt', 'lr', 'svm', 'knn']:
        raise Exception('Wrong model name')
    
    # Split data based on time or number of events
    split_type = sys.argv[3]
    if split_type not in ['time', 'events']:
        raise Exception('Wrong split type')


    print(f'PARAMS: {dataset} {model_name} {split_type}')

    # ─────────────────────────────────────────────
    # READ DATA
    # ─────────────────────────────────────────────

    print(f"[START] READ DATA - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # Read training and testing data
    x_train, y_train = read_data(dataset, split_type, 'train')
    x_val, y_val = read_data(dataset, split_type, 'val')
    x_test, y_test = read_data(dataset, split_type, 'test')

    print(f'{len(x_train)}, {len(y_train)}')
    print(f'{len(x_val)}, {len(y_val)}')
    print(f'{len(x_test)}, {len(y_test)}')

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

    print(f"[END] READ DATA - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

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
    elif model_name == 'dt':
        model_constructor = DecisionTreeClassifier
        param_dist = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20, 40],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': [None, 'sqrt', 'log2'],
            'min_impurity_decrease': [0.0, 0.001, 0.01],
            'ccp_alpha': [0.0, 0.001, 0.01],
            'class_weight': [None, 'balanced', {0:1, 1:3}, {0:1, 1:6}, {0:1, 1:10}],
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
    elif model_name == 'svm':
        model_constructor = SVC
        param_dist = {
            'C': np.logspace(-2, 2, 9),
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 7)),
            'class_weight': [None, 'balanced', {0:1, 1:3}, {0:1, 1:6}, {0:1, 1:10}],
            'probability': [True],
            'max_iter': [2000, 5000],
            'tol': [1e-4, 1e-3],
            # do not include n_jobs or random_state; SVC doesn't support them
        }
    elif model_name == 'knn':
        model_constructor = KNeighborsClassifier
        param_dist = {
            'n_neighbors': list(range(3, 31, 2)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'kd_tree', 'ball_tree'],
            'p': [1, 2],
            'metric': ['euclidean', 'manhattan'],
            'leaf_size': [20, 30, 40, 50],
            'n_jobs': [-1],
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

    results_base = f"results_baseline/{split_type}/{dataset}"
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