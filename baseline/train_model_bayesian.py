from find_motifs import *
import os
import sys
import numpy as np
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
import optuna
from optuna.samplers import TPESampler


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
    tpe_sampler = TPESampler(seed=RANDOM_STATE)

    # dataset = 'email-Enron'
    # model_name = 'lr'
    n_iter = 1000 # Number of random combinations to try for hyperparameters


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
    # lr - Logistic Regression
    model_name = sys.argv[2]
    if model_name not in ['rf', 'xgb', 'lr']:
        raise Exception('Wrong model name')
    
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

    def objective(trial):
        if model_name == "rf":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
                max_depth=trial.suggest_int("max_depth", 10, 40) if trial.suggest_categorical("use_depth_limit", [True, False]) else None,
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ['sqrt', 'log2', 0.5, 0.7, 1.0]),
                bootstrap=True,  # always True
                oob_score=trial.suggest_categorical("oob_score", [True, False]),
                max_samples=trial.suggest_float("max_samples", 0.7, 1.0) if trial.suggest_categorical("use_sample_limit", [True, False]) else None,
                min_impurity_decrease=trial.suggest_float("min_impurity_decrease", 0.0, 0.01),
                criterion=trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss']),
                ccp_alpha=trial.suggest_float("ccp_alpha", 0.0, 0.01),
                class_weight=trial.suggest_categorical("class_weight", [None, 'balanced', 'balanced_subsample']),
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        elif model_name == "xgb":
            model = xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                gamma=trial.suggest_float("gamma", 0, 3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                colsample_bynode=trial.suggest_float("colsample_bynode", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.01, 50),
                scale_pos_weight=trial.suggest_int("scale_pos_weight", 1, 20),
                n_jobs=-1,
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            )
        elif model_name == "lr":
            model = LogisticRegression(
                penalty=trial.suggest_categorical("penalty", ['l2', 'l1']),
                C=trial.suggest_float("C", 1e-4, 1e4,log=True),
                solver=trial.suggest_categorical("solver", ['liblinear', 'saga']),
                max_iter=trial.suggest_int("max_iter", 200, 1000, step=100),
                fit_intercept=trial.suggest_categorical("fit_intercept", [True, False]),
                class_weight=trial.suggest_categorical("class_weight", [None, 'balanced', {0:1,1:2},{0:1,1:3},{0:1,1:5},{0:1,1:10}]),
                random_state=RANDOM_STATE
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented in Optuna objective.")

        # Train model
        model.fit(x_train_scaled, y_train)

        # Evaluate on validation set
        y_val_pred = model.predict_proba(x_val_scaled)[:, 1]
        avg_prec = average_precision_score(y_val, y_val_pred)
        auc_score = roc_auc_score(y_val, y_val_pred)
        performance = avg_prec / (sum(y_val) / len(y_val)) # avg_prec / random_baseline

        best_score = 0
        try:
            best_score = trial.study.best_value
        except ValueError:
            pass

        print(f"Iteration {trial.number+1}/{n_iter}: Score = {performance:.4f}, Best = {best_score:.4f}")

        return performance

    study = optuna.create_study(direction='maximize', sampler=tpe_sampler)
    study.optimize(objective, n_trials=n_iter)

    best_score = study.best_value
    best_params = study.best_params

    _ =  {key: best_params.pop(key) for key in ["use_depth_limit", "use_sample_limit"] if key in best_params}

    print("Best parameters:", best_params)
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
    if model_name == "rf":
        model = RandomForestClassifier()
    elif model_name == "xgb":
        model = xgb.XGBClassifier()
    elif model_name == "lr":
        model = LogisticRegression()
    else:
        raise Exception('Wrong model name')
    
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

    results_base = f"results_baseline_bayesian/{split_type}/{dataset}"
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