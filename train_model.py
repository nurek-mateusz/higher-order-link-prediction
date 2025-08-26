import compute_features as cf
import os
import sys
import numpy as np
import random
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


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
if dataset not in ['coauth-DBLP', 'coauth-MAG-Geology', 'coauth-MAG-History', 'congress-bills', 'contact-high-school',
                   'contact-primary-school', 'DAWN', 'email-Enron', 'email-Eu', 'NDC-classes', 'NDC-substances', 'tags-ask-ubuntu', 
                   'tags-math-sx', 'tags-stack-overflow', 'threads-ask-ubuntu', 'threads-math-sx', 'threads-stack-overflow']:
    raise Exception('Wrong dataset name')

model_name = sys.argv[2]
if model_name not in ['rf', 'xgb', 'dt', 'lr', 'svm', 'knn']:
    raise Exception('Wrong model name')


# ─────────────────────────────────────────────
# SPLIT DATA
# ─────────────────────────────────────────────

# Read data
with open(f'data/raw/{dataset}/{dataset}-simplices.txt', 'r') as file:
    verts = [int(line.strip()) for line in file]

with open(f'data/raw/{dataset}/{dataset}-nverts.txt', 'r') as file:
    nverts = [int(line.strip()) for line in file]

with open(f'data/raw/{dataset}/{dataset}-times.txt', 'r') as file:
    times = [int(line.strip()) for line in file]

# Create simplices
simplices = []
index = 0
for count in nverts:
    simplex = verts[index:index+count]
    simplices.append(simplex)
    index += count

# Combine simplices with their timestamps and sort them chronologically
combined = list(zip(simplices, nverts, times))
combined.sort(key=lambda x: x[2])

# Remove simplices with fewer than two vertices
combined = [x for x in combined if x[1] >= 2]

# Split the data into training and test sets based on time
min_time = combined[0][2]
max_time = combined[-1][2]
threshold_60 = min_time + round((max_time - min_time) * 0.6)
threshold_70 = min_time + round((max_time - min_time) * 0.7)
threshold_80 = min_time + round((max_time - min_time) * 0.8)

# 0 - 60% of data: features for training set
# 60% - 80% of data: labels for training set
#
# 0 - 70% of data: features for validation set
# 70% - 80% of data: labels for validation set
#
# 0% - 80% of data: features for test set
# 80% - 100% of data: labels for test set
data_0_60 = [x for x in combined if x[2] <= threshold_60]
data_60_80 = [x for x in combined if (x[2] > threshold_60) & (x[2] <= threshold_80)]

data_0_70 = [x for x in combined if x[2] <= threshold_70]
data_70_80 = [x for x in combined if (x[2] > threshold_70) & (x[2] <= threshold_80)]

data_0_80 = [x for x in combined if x[2] <= threshold_80]
data_80_100 = [x for x in combined if x[2] > threshold_80]

def unpack(data):
    verts = []
    nverts = []
    times = []
    for v, n, t in data:
        verts.extend(v)
        nverts.append(n)
        times.append(t)
    return verts, nverts, times

simplices_0_60, nverts_0_60, times_0_60 = unpack(data_0_60)
simplices_60_80, nverts_60_80, times_60_80 = unpack(data_60_80)

simplices_0_70, nverts_0_70, times_0_70 = unpack(data_0_70)
simplices_70_80, nverts_70_80, times_70_80 = unpack(data_70_80)

simplices_0_80, nverts_0_80, times_0_80 = unpack(data_0_80)
simplices_80_100, nverts_80_100, times_80_100 = unpack(data_80_100)

dataset_0_60 = {'nverts': nverts_0_60, 'simplices': simplices_0_60, 'times': times_0_60}
dataset_60_80 = {'nverts': nverts_60_80, 'simplices': simplices_60_80, 'times': times_60_80}
dataset_0_70 = {'nverts': nverts_0_70, 'simplices': simplices_0_70, 'times': times_0_70}
dataset_70_80 = {'nverts': nverts_70_80, 'simplices': simplices_70_80, 'times': times_70_80}
dataset_0_80 = {'nverts': nverts_0_80, 'simplices': simplices_0_80, 'times': times_0_80}

# Create candidate open triangles for training set
generator_0_60 = cf.DataPreparation(n_workers=10)
generator_0_60.build_data_structures(dataset_0_60)
candidates_0_60 = generator_0_60.generate_candidate_triangles()

# Create candidate open triangles for validation set
generator_0_70 = cf.DataPreparation(n_workers=10)
generator_0_70.build_data_structures(dataset_0_70)
candidates_0_70 = generator_0_70.generate_candidate_triangles()

# Create candidate open triangles for test set
generator_0_80 = cf.DataPreparation(n_workers=10)
generator_0_80.build_data_structures(dataset_0_80)
candidates_0_80 = generator_0_80.generate_candidate_triangles()


# ─────────────────────────────────────────────
# CREATE FEATURES
# ─────────────────────────────────────────────

features_0_60 = generator_0_60.calculate_triangle_features(candidates_0_60)
features_0_70 = generator_0_70.calculate_triangle_features(candidates_0_70)
features_0_80 = generator_0_80.calculate_triangle_features(candidates_0_80)

feature_names = ['intensity', 'lifetime', 'internal_density', 'structural_imbalance', 'reinforcement']

# Train set features
x_train = []
for i, item in enumerate(features_0_60):
    features = [item[feature] for feature in feature_names]
    x_train.append(features)
x_train = np.array(x_train)

# Validation set features
x_val = []
for i, item in enumerate(features_0_70):
    features = [item[feature] for feature in feature_names]
    x_val.append(features)
x_val = np.array(x_val)

# Test set features
x_test = []
for item in features_0_80:
    features = [item[feature] for feature in feature_names]
    x_test.append(features)
x_test = np.array(x_test)

# Train set labels
simplices_60_80 = {tuple(sorted(x)) for x,_,_ in data_60_80}
y_train = [1 if x in simplices_60_80 else 0 for x in candidates_0_60]
y_train = np.array(y_train)

# Validation set labels
simplices_70_80 = {tuple(sorted(x)) for x,_,_ in data_70_80}
y_val = [1 if x in simplices_70_80 else 0 for x in candidates_0_70]
y_val = np.array(y_val)

# Test set labels
simplices_80_100 = {tuple(sorted(x)) for x,_,_ in data_80_100}
labels_0_80 = [1 if x in simplices_80_100 else 0 for x in candidates_0_80]
y_test = np.array(labels_0_80)

def under_sample(x_train, y_train, ratio=1):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
    x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
    return x_resampled, y_resampled

print(f'{len(x_train)}, {len(y_train)}')
print(f'{len(x_val)}, {len(y_val)}')
print(f'{len(x_test)}, {len(y_test)}')

print(f'Before undersampling train set: zeros: {np.count_nonzero(y_train == 0)}, ones: {np.count_nonzero(y_train == 1)}')

# Negative edge undersampling
x_train, y_train = under_sample(x_train, y_train, ratio=0.33)

print(f'After undersampling train set: zeros: {np.count_nonzero(y_train == 0)}, ones: {np.count_nonzero(y_train == 1)}')


# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

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
best_model = None

# Perform random search
for i in range(n_iter):
    # Randomly sample parameters
    params = {key: random.choice(values) for key, values in param_dist.items()}
    
    # Create and train model
    model = model_constructor()
    model.set_params(**params)
    model.fit(x_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict_proba(x_val)[:, 1]
    avg_prec = average_precision_score(y_val, y_val_pred)
    auc_score = roc_auc_score(y_val, y_val_pred)
    performance = avg_prec / (sum(y_val) / len(y_val)) # avg_prec / random_baseline
    
    # Set which metric to use to choose the best model
    score = performance
    
    # Track best model
    if score > best_score:
        best_score = score
        best_params = params
        best_model = model
        
    print(f"Iteration {i+1}/{n_iter}: Score = {score:.4f}, Best = {best_score:.4f}")

print("\nBest parameters:", best_params)
print("Best validation score:", best_score)

# Fin final model
model = model_constructor()
model.set_params(**best_params)
model.fit(x_val, y_val)

# Final evaluation on test set
y_test_pred = model.predict_proba(x_test)[:, 1]
avg_prec = average_precision_score(y_test, y_test_pred)
auc_score = roc_auc_score(y_test, y_test_pred)
performance = avg_prec / (sum(y_test) / len(y_test)) # avg_prec / random_baseline

auc_score = roc_auc_score(y_test, y_test_pred)
print('Test set:', 'Performance', round(performance, 4), 'AP', round(avg_prec, 4), 'AUC', round(auc_score, 4))


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────

# Prepare directiories
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

results_base = f"results/{dataset}"
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
    pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)