import compute_features as cf
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


def compute_composite_indices(x, alpha, beta, gamma, w1, w2, w3, w4, w5):
    INTENSITY = 0
    LIFETIME = 1
    INTERNAL_DENSITY = 2
    STRUCTURAL_IMBALANCE = 3
    REINFORCEMENT = 4

    twdi = (alpha * x[:,INTERNAL_DENSITY] + beta * x[:,REINFORCEMENT]) * np.exp(-gamma * np.abs(x[:,LIFETIME]))

    reinforcement_boost = np.tanh(beta * x[:,REINFORCEMENT])
    lifetime_factor = 1 / (1 + gamma * np.abs(x[:,LIFETIME]))
    hai = x[:,INTERNAL_DENSITY] * (1 + reinforcement_boost) * lifetime_factor

    ddi = (x[:,INTERNAL_DENSITY] * x[:,REINFORCEMENT]) / np.sqrt(np.abs(x[:,LIFETIME]) + 1)

    wci = (w1 * x[:,INTENSITY] + 
            w2 * x[:,LIFETIME] + 
            w3 * x[:,INTERNAL_DENSITY] + 
            w4 * x[:,STRUCTURAL_IMBALANCE] +
            w5 * x[:,REINFORCEMENT])
    
    return twdi, hai, ddi, wci

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
    
    # Split data based on time or number of events
    split_type = sys.argv[3]
    if split_type not in ['time', 'events']:
        raise Exception('Wrong split type')


    print(f'PARAMS: {dataset} {model_name} {split_type}')

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
        y_train = np.array(y_train)

    with open(f'processing_dataset/{split_type}/{dataset}/y_val.pickle', 'rb') as file:
        y_val = pickle.load(file)
        y_val = np.array(y_val)

    with open(f'processing_dataset/{split_type}/{dataset}/y_test.pickle', 'rb') as file:
        y_test = pickle.load(file)
        y_test = np.array(y_test)

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

    # Convert features to numpy format
    feature_names = ['intensity', 'lifetime', 'internal_density', 'structural_imbalance', 'reinforcement']

    # features = {
    #     'triangle': triangle,
    #     'intensity': calculate_intensity(triangle, simplices, node_to_times, node_time_to_neighbors),
    #     'lifetime': calculate_lifetime(triangle, edge_times),
    #     'internal_density': calculate_internal_density(triangle, pair_to_times),
    #     'structural_imbalance': calculate_structural_imbalance(triangle, pair_to_times),
    #     'reinforcement': calculate_reinforcement(triangle, node_to_simplices)
    # }

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

    print(f'{len(x_train)}, {len(y_train)}')
    print(f'{len(x_val)}, {len(y_val)}')
    print(f'{len(x_test)}, {len(y_test)}')

    # Negative edge undersampling
    def under_sample(x_train, y_train, ratio=1):
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
        x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
        return x_resampled, y_resampled
    
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
        
        # Aditional learnable weights
        alpha = trial.suggest_float("w_alpha", 0, 3)
        beta = trial.suggest_float("w_beta", 0, 3)
        gamma = trial.suggest_float("w_gamma", 1e-4, 3, log=True)
        w1 = trial.suggest_float("w1", 0, 3)
        w2 = trial.suggest_float("w2", 0, 3)
        w3 = trial.suggest_float("w3", 0, 3)
        w4 = trial.suggest_float("w4", 0, 3)
        w5 = trial.suggest_float("w5", 0, 3)

        # Compute composite indices
        twdi_train, hai_train, ddi_train, wci_train = compute_composite_indices(x_train_scaled, alpha, beta, gamma, w1, w2, w3, w4, w5)
        twdi_val, hai_val, ddi_val, wci_val = compute_composite_indices(x_val_scaled, alpha, beta, gamma, w1, w2, w3, w4, w5)

        # Add to existing features
        x_train_comb = np.column_stack([x_train_scaled, twdi_train, ddi_train, wci_train])
        x_val_comb = np.column_stack([x_val_scaled, twdi_val, ddi_val, wci_val])

        # Train model
        model.fit(x_train_comb, y_train)

        # Evaluate on validation set
        y_val_pred = model.predict_proba(x_val_comb)[:, 1]
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
    weight_names = ["w_alpha", "w_beta", "w_gamma", "w1", "w2", "w3", "w4", "w5"]
    composite_weights = {key: best_params.pop(key) for key in weight_names if key in best_params}

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

    alpha = composite_weights["w_alpha"]
    beta = composite_weights["w_beta"]
    gamma = composite_weights["w_gamma"]
    w1 = composite_weights["w1"]
    w2 = composite_weights["w2"]
    w3 = composite_weights["w3"]
    w4 = composite_weights["w4"]
    w5 = composite_weights["w5"]

    twdi_val, hai_val, ddi_val, wci_val = compute_composite_indices(x_val, alpha, beta, gamma, w1, w2, w3, w4, w5)
    twdi_test, hai_test, ddi_test, wci_test = compute_composite_indices(x_test, alpha, beta, gamma, w1, w2, w3, w4, w5)

    x_val = np.column_stack([x_val, twdi_val, hai_val, ddi_val, wci_val])
    x_test = np.column_stack([x_test, twdi_test, hai_test, ddi_test, wci_test])

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

    results_base = f"results_bayesian/{split_type}/{dataset}"
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
    best_params["w_alpha"] = alpha
    best_params["w_beta"] = beta
    best_params["w_gamma"] = gamma
    best_params["w1"] = w1
    best_params["w2"] = w2
    best_params["w3"] = w3
    best_params["w4"] = w4
    best_params["w5"] = w5
    with open(f"{params_dir}/best_params_{model_name}.txt", "w") as f:
        f.write(str(best_params))

    # Save best model
    with open(f"{model_dir}/best_model_{model_name}.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[END] SAVE RESULTS - {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")