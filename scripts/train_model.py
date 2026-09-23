import prepare_features as pf
import os
import sys
import numpy as np
import pandas as pd
import random
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from itertools import product


# Negative edge undersampling
def under_sample(x_train, y_train, ratio, seed):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
    return x_resampled, y_resampled


if __name__ == "__main__":
    # dataset = 'email-Enron'
    # model_name = 'lr'
    # feature_type = 'a'

    n_iter = 500 # Number of random combinations to try for hyperparameters

    # ─────────────────────────────────────────────
    # PARSE ARGUMENTS
    # ─────────────────────────────────────────────

    if len(sys.argv) != 4:
        raise Exception('Wrong number of arguments')

    dataset = sys.argv[1]
    if dataset not in ['coauth-MAG-Geology', 'coauth-MAG-History', 'contact-high-school', 'contact-primary-school',
                       'email-Enron', 'email-Eu', 'NDC-classes', 'NDC-substances', 'tags-ask-ubuntu', 'threads-ask-ubuntu']:
        raise Exception('Wrong dataset name')

    # rf - Random Forest
    # xgb - XGBoost
    # lr - Logistic Regression
    model_name = sys.argv[2]
    if model_name not in ['rf', 'xgb', 'lr']:
        raise Exception('Wrong model name')
    
    # w - weight features
    # d - degree features
    # h - heuristics features
    # t - time features
    # m - motif features
    # a - all features (weight + degree + heuristics + time + motif)
    feature_type = sys.argv[3]
    if feature_type not in ['w', 'd', 'h', 't', 'm', 'a', 'a-w', 'a-d', 'a-h', 'a-t', 'a-m']:
        raise Exception('Wrong feature_type value')

    n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))

    print(f'PARAMS: {dataset} {model_name} {feature_type} {n_cores}')

    # ─────────────────────────────────────────────
    # PREPARE DIRECTORIES AND FILES TO SAVE RESULTS
    # ─────────────────────────────────────────────
    results_base = f"../results/{dataset}"
    params_dir = f"{results_base}/best_params"
    metrics_dir = f"{results_base}/metrics"

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    metrics_file = f'{metrics_dir}/{model_name}_{feature_type}.csv'
    params_file = f'{params_dir}/{model_name}_{feature_type}.csv'

    for f in [metrics_file, params_file]:
        if os.path.exists(f):
            os.remove(f)

    # ─────────────────────────────────────────────
    # MAIN TRAINING LOOP
    # ─────────────────────────────────────────────
    SEEDS = [3248, 5941, 6553, 8477, 255]
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        
        # ─────────────────────────────────────────────
        # CREATE FEATURES
        # ─────────────────────────────────────────────
        x_train, y_train, feature_names = pf.load_features(dataset, 'train', feature_type)
        x_val, y_val, _ = pf.load_features(dataset, 'val', feature_type)
        x_test, y_test, _ = pf.load_features(dataset, 'test', feature_type)
        
        print(f'x_train.shape={x_train.shape}, y_train.shape={y_train.shape}')
        print(f'x_val.shape={x_val.shape}, y_val.shape={y_val.shape}')
        print(f'x_test.shape={x_test.shape}, y_test.shape={y_test.shape}')

        # ─────────────────────────────────────────────
        # NORMALIZATION AND UNDERSAMPLING
        # ─────────────────────────────────────────────

        # Nromalize features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        print(f'Before undersampling train set: zeros: {np.count_nonzero(y_train == 0)}, ones: {np.count_nonzero(y_train == 1)}')

        try:
            x_train_scaled, y_train = under_sample(x_train_scaled, y_train, ratio=0.33, seed=seed)
        except ValueError as e:
            # IF ValueError: The specified ratio required to generate new sample in the majority class while trying to remove samples. Please increase the ratio.
            # DO NOTHING - undersampling is not necessery because class are more or less equal
            print(e)

        print(f'After undersampling train set: zeros: {np.count_nonzero(y_train == 0)}, ones: {np.count_nonzero(y_train == 1)}')

        # ─────────────────────────────────────────────
        # TRAIN MODEL
        # ─────────────────────────────────────────────

        print(f"[START] TRAIN MODEL - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

        # Hyperparameters
        neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        if model_name == 'rf':
            model_constructor = RandomForestClassifier
            param_dist = {
                'n_estimators': [500],
                'max_depth': [None, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10, 20, 50],
                'max_features': ['sqrt', 0.2, 0.3, 0.5, 0.7],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
                'criterion': ['gini', 'entropy'],
                'n_jobs': [n_cores],
                'random_state': [seed],
            }
        elif model_name == 'xgb':
            model_constructor = xgb.XGBClassifier
            param_dist = {
                'n_estimators': [2000],
                'early_stopping_rounds': [100],
                'learning_rate': [0.03, 0.05, 0.1],
                'max_depth': [2, 3, 4, 6, 8],
                'min_child_weight': [1, 5, 20, 50],
                'subsample': [0.7, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.75, 1.0],
                'reg_lambda': [1, 10, 50],
                'reg_alpha': [0, 1],
                'scale_pos_weight': [1, np.sqrt(neg_pos_ratio), neg_pos_ratio],
                'eval_metric': ['logloss'],
                'n_jobs': [n_cores],
                'random_state': [seed],
            }
        elif model_name == 'lr':
            model_constructor = LogisticRegression
            param_dist = {
                'l1_ratio': [0, 1],
                'C': np.logspace(-4, 4, 40),
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [5000],
                'fit_intercept': [True, False],
                'class_weight': [None, 'balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:5}, {0:1, 1:10}],
                'random_state': [seed],
            }

        best_score = -np.inf
        best_params = None

        # Perform random search
        # Randomly sample parameters
        keys = list(param_dist.keys())
        all_combos = [dict(zip(keys, v)) for v in product(*param_dist.values())]
        if model_name == 'lr':
            all_combos = [c for c in all_combos
                        if not (c['solver'] == 'lbfgs' and c['l1_ratio'] == 1)]

        if len(all_combos) <= n_iter:
            combos = all_combos
        else:
            rng = random.Random(seed)
            combos = rng.sample(all_combos, n_iter)

        # Perform random search
        for i, params in enumerate(combos):          
            # Create and train model
            model = model_constructor()
            model.set_params(**params)
            
            if model_name == 'xgb':
                # for early stopping
                model.fit(x_train_scaled, y_train, eval_set=[(x_val_scaled, y_val)], verbose=False)
            else:
                model.fit(x_train_scaled, y_train)
            
            # Evaluate on validation set
            y_val_pred = model.predict_proba(x_val_scaled)[:, 1]
            avg_prec = average_precision_score(y_val, y_val_pred)
            auc_score = roc_auc_score(y_val, y_val_pred)
            performance = avg_prec / np.mean(y_val) # avg_prec / random_baseline
            
            # Set which metric to use to choose the best model
            score = performance
            
            # Track best model
            if score > best_score:
                best_score = score
                best_params = dict(params)
                if model_name == 'xgb':
                    best_params['n_estimators'] = model.best_iteration + 1
                    best_params.pop('early_stopping_rounds', None)
                
            print(f"Iteration {i+1}/{len(combos)}: Score = {score:.4f}, Best = {best_score:.4f}")

        print("\nBest parameters:", best_params)
        print("Best validation score:", best_score)

        # Final undersampling and data normalization
        scaler = StandardScaler()
        x_val = scaler.fit_transform(x_val)
        x_test = scaler.transform(x_test)
        try:
            x_val, y_val = under_sample(x_val, y_val, ratio=0.33, seed=seed)
        except ValueError as e:
            # IF ValueError: The specified ratio required to generate new sample in the majority class while trying to remove samples. Please increase the ratio.
            # DO NOTHING - undersampling is not necessery because class are more or less equal
            print(e)

        # Fit final model
        model = model_constructor()
        model.set_params(**best_params)
        model.fit(x_val, y_val)

        # Final evaluation on test set
        y_test_pred = model.predict_proba(x_test)[:, 1]

        avg_prec = average_precision_score(y_test, y_test_pred)
        performance = avg_prec / np.mean(y_test) # avg_prec / random_baseline
        auc_score = roc_auc_score(y_test, y_test_pred)

        print('Test set:', 'Performance', round(performance, 4), 'AP', round(avg_prec, 4), 'AUC', round(auc_score, 4))

        print(f"[END] TRAIN MODEL - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

        # ─────────────────────────────────────────────
        # SAVE RESULTS
        # ─────────────────────────────────────────────
        row = {
            'seed': seed,
            'performance': performance,
            'avg_prec': avg_prec,
            'auc': auc_score,
        }
        
        pd.DataFrame([row]).to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file), index=False)
        pd.DataFrame([{'seed': seed, **best_params}]).to_csv(params_file, mode='a', header=not os.path.exists(params_file), index=False)

