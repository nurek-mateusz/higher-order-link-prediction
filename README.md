# higher-order-link-prediction

# How to Run
## Dataset split
Run `python read_simplices_data.py <SPLIT_TYPE>`

where `<SPLIT_TYPE>` is `time` (split based on time) or `events` (split based on number of events)

## Model training
Run `python train_model.py <DATASET> <MODEL_NAME> <SPLIT_TYPE> <WITH_MOTIFS>`

where:
- `<DATASET>` is one of `coauth-MAG-Geology`, `coauth-MAG-History`, `contact-high-school`, `contact-primary-school`, `contact-primary-school-2`, `email-Enron`, `email-Eu`, `NDC-classes`, `NDC-substances`, `tags-ask-ubuntu`, `threads-ask-ubuntu`
- `<MODEL_NAME>` is one of `rf` (Random Forest), `xgb` (XGBoost), `lr` (Logistic Regression)
- `<SPLIT_TYPE>` is `time` (split based on time) or `events` (split based on number of events)
- `<WITH_MOTIFS>` is `y` (our features + all [motif features from the Rongmei article](https://www.sciencedirect.com/science/article/abs/pii/S0957417424031518)) or `n` (only our features)

## Baseline model
Go to the `baseline` directory.

### Compute motif features
Run `python construct_motif_feature.py <DATASET> <SPLIT_TYPE>`

where:
- `<DATASET>` is one of `coauth-MAG-Geology`, `coauth-MAG-History`, `contact-high-school`, `contact-primary-school`, `contact-primary-school-2`, `email-Enron`, `email-Eu`, `NDC-classes`, `NDC-substances`, `tags-ask-ubuntu`, `threads-ask-ubuntu`
- `<SPLIT_TYPE>` is `time` (split based on time) or `events` (split based on number of events)

### Baseline model training
Run the script `python train_model.py` in `baseline` directory similar to the above *Model training* description

# Results
Results obtained for the test set are saved in:
- Our model: `results_our/<SPLIT_TYPE>/<DATASET>/metrics/test_<MODEL_NAME>.csv` and `results_our_and_motifs/<SPLIT_TYPE>/<DATASET>/metrics/test_<MODEL_NAME>.csv`
- Baseline model: `baseline/results_baseline/<SPLIT_TYPE>/<DATASET>/metrics/test_<MODEL_NAME>.csv`

# Additional scripts:
- `compute_features.py` used by `train_model.py` to compute our features
- `baseline\find_motifs.py` used by `\baseline\construct_motif_feature.py`
- `create_plots.ipynb` creates barplots with results
- `train_model_bayesian.py` use to train our model with composite indices


