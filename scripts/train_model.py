import prepare_features as pf
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
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    # feature_type = 'o'
    # n_motifs = 75

    n_iter = 500 # Number of random combinations to try for hyperparameters

    # ─────────────────────────────────────────────
    # PARSE ARGUMENTS
    # ─────────────────────────────────────────────

    if len(sys.argv) not in [4,5]:
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
    
    # o - our features
    # m - motif features
    # b - both, our and motif features
    feature_type = sys.argv[3]
    if feature_type not in ['o', 'm', 'b']:
        raise Exception('Wrong feature_type value')
    
    if len(sys.argv) == 4   :
        n_motifs = 75
    else:
        n_motifs = sys.argv[4]
        try:
            n_motifs = int(n_motifs)
        except ValueError:
            raise Exception("n_motifs is not an integer")

    if n_motifs not in [1, 3, 5, 10, 20, 40, 75]:
        raise Exception("The n_motifs must be in [1, 3, 5, 10, 20, 40, 75]")

    print(f'PARAMS: {dataset} {model_name} {feature_type} {n_motifs}')

    # ─────────────────────────────────────────────
    # CREATE FEATURES
    # ─────────────────────────────────────────────

    print(f"[START] CREATE FEATURES - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    n_cores = os.cpu_count()

    if feature_type == 'o':
        x_train, y_train = pf.create_our_features(dataset, 'train')
        x_val, y_val = pf.create_our_features(dataset, 'val')
        x_test, y_test = pf.create_our_features(dataset, 'test')
    elif feature_type == 'm':
        x_train, y_train = pf.create_motif_features(dataset, model_name, 'train', n_motifs, feature_type)
        x_val, y_val = pf.create_motif_features(dataset, model_name, 'val', n_motifs, feature_type)
        x_test, y_test = pf.create_motif_features(dataset, model_name, 'test', n_motifs, feature_type)
    else:
        x_train, y_train = pf.create_our_and_motif_features(dataset, model_name, 'train', n_motifs, feature_type)
        x_val, y_val = pf.create_our_and_motif_features(dataset, model_name, 'val', n_motifs, feature_type)
        x_test, y_test = pf.create_our_and_motif_features(dataset, model_name, 'test', n_motifs, feature_type)

    # Save feature names
    feature_names = [col for col in x_test.columns if col != 'triangle']

    # Convert to numpy
    x_train, y_train = pf.convert_to_numpy_array(x_train, y_train)
    x_val, y_val = pf.convert_to_numpy_array(x_val, y_val)
    x_test, y_test = pf.convert_to_numpy_array(x_test, y_test)
    
    print(f'x_train.shape={x_train.shape}, y_train.shape={y_train.shape}')
    print(f'x_val.shape={x_val.shape}, y_val.shape={y_val.shape}')
    print(f'x_test.shape={x_test.shape}, y_test.shape={y_test.shape}')
    
    print(f"[END] CREATE FEATURES - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────
    # UNDERSAMPLING AND NORMALIZATION
    # ─────────────────────────────────────────────

    print(f"[START] UNDERSAMPLING AND NORMALIZATION - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

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

    print(f"[END] UNDERSAMPLING AND NORMALIZATION - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

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
    performance = avg_prec / (sum(y_test) / len(y_test)) # avg_prec / random_baseline
    auc_score = roc_auc_score(y_test, y_test_pred)

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

    if feature_type == 'o':
        feature_set = 'our'
        best_motifs = ''
    elif feature_type == 'm':
        feature_set = 'motifs'
        best_motifs = f'_{n_motifs}' # Add information how many motif features have been used
    else:
        feature_set = 'our_and_motifs'
        best_motifs = f'_{n_motifs}' # Add information how many motif features have been used

    results_base = f"../results/{feature_set}/{dataset}"
    model_dir = f"{results_base}/best_model"
    params_dir = f"{results_base}/best_params"
    metrics_dir = f"{results_base}/metrics"

    for dir in [model_dir, params_dir, metrics_dir]:
        ensure_dir(dir)

    # Save metrics
    with open(f"{metrics_dir}/test_{model_name}{best_motifs}.csv", "w") as f:
        f.write("performance,avg_prec,auc\n")
        f.write(f"{performance},{avg_prec},{auc_score}\n")

    # Save best params
    with open(f"{params_dir}/best_params_{model_name}{best_motifs}.txt", "w") as f:
        f.write(str(best_params))

    # Save best model
    with open(f"{model_dir}/best_model_{model_name}{best_motifs}.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[END] SAVE RESULTS - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # ─────────────────────────────────────────────
    # SHAP ANALYSIS
    # ─────────────────────────────────────────────

    print(f"[START] SHAP - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
        
    def plot_shap(shap_obj, title, X_data, out_path):
        plt.figure()
        plt.title(title)
        shap.summary_plot(shap_obj, X_data, show=False)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    def save_shap_ranking(shap_obj, feature_names, out_path):
        # shap_obj can be a shap.Explanation or a raw numpy array
        if hasattr(shap_obj, 'values'):
            vals = shap_obj.values
        else:
            vals = shap_obj

        mean_abs = np.abs(vals).mean(axis=0)

        df = pd.DataFrame({
            'feature_name': feature_names,
            'mean_abs_shap': mean_abs
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        df['rank'] = df.index + 1
        df.to_csv(out_path, index=False)

    def subsample_for_shap(X, y, n, seed):
        if len(X) <= n:
            return X
        keep, _ = train_test_split(np.arange(len(X)), train_size=n, stratify=y, random_state=seed)
        keep = np.sort(keep)
        return X.iloc[keep] if hasattr(X, 'iloc') else X[keep]

    shap_plot_path = f'{metrics_dir}/shap_{model_name}{best_motifs}.pdf'
    shap_ranking_path = f'{metrics_dir}/shap_ranking_{model_name}{best_motifs}.csv'

    SHAP_N = 10000
    SHAP_SEED = 0
    x_shap = subsample_for_shap(x_test, y_test, SHAP_N, SHAP_SEED)

    if model_name == 'xgb':
        explainer_xgb = shap.TreeExplainer(model)
        shap_values_xgb = explainer_xgb(x_shap, check_additivity=False)
        shap_values_xgb.feature_names = feature_names
        plot_shap(shap_values_xgb, "XGBoost", x_shap, shap_plot_path)
        save_shap_ranking(shap_values_xgb, feature_names, shap_ranking_path)

    if model_name == 'rf':
        explainer_rf = shap.TreeExplainer(model)
        shap_values_rf = explainer_rf(x_shap, check_additivity=False)
        shap_values_rf = shap_values_rf[:, :, 1]
        shap_values_rf.feature_names = feature_names
        plot_shap(shap_values_rf, "Random Forest", x_shap, shap_plot_path)
        save_shap_ranking(shap_values_rf, feature_names, shap_ranking_path)

    if model_name == 'lr':
        explainer_lr = shap.LinearExplainer(
            model, shap.maskers.Independent(x_val, max_samples=1000))
        shap_values_lr = explainer_lr(x_shap)
        shap_values_lr.feature_names = feature_names
        plot_shap(shap_values_lr, "Logistic Regression", x_shap, shap_plot_path)
        save_shap_ranking(shap_values_lr, feature_names, shap_ranking_path)
    
    print(f"[END] SHAP - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    