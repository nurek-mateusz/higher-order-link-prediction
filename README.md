# higher-order-link-prediction

# How to Run
## Dataset split
Run `python read_simplices_data.py <SPLIT_TYPE>`

where `<SPLIT_TYPE>` is `time` (split based on time) or `events` (split based on number of events)

## Compute motif features
Go to the `baseline` directory.
Run `python construct_motif_feature.py <DATASET> <SPLIT_TYPE>`

where:
- `<DATASET>` is one of `coauth-MAG-Geology`, `coauth-MAG-History`, `contact-high-school`, `contact-primary-school`, `contact-primary-school-2`, `email-Enron`, `email-Eu`, `NDC-classes`, `NDC-substances`, `tags-ask-ubuntu`, `threads-ask-ubuntu`
- `<SPLIT_TYPE>` is `time` (split based on time) or `events` (split based on number of events)

## Model training
Run `python train_model.py <DATASET> <MODEL_NAME> <SPLIT_TYPE> <FEATURE_TYPE> <N_MOTIFS>`

where:
- `<DATASET>` is one of `coauth-MAG-Geology`, `coauth-MAG-History`, `contact-high-school`, `contact-primary-school`, `contact-primary-school-2`, `email-Enron`, `email-Eu`, `NDC-classes`, `NDC-substances`, `tags-ask-ubuntu`, `threads-ask-ubuntu`
- `<MODEL_NAME>` is one of `rf` (Random Forest), `xgb` (XGBoost), `lr` (Logistic Regression)
- `<SPLIT_TYPE>` is `time` (split based on time) or `events` (split based on number of events)
- `<FEATURE_TYPE>` is one of `o` (our features), `m` [motif features from the Rongmei article](https://www.sciencedirect.com/science/article/abs/pii/S0957417424031518)), `b` (both our + motif features)
- `<N_MOTIFS>` is one of `75, 40, 20, 10, 5, 3, 1`. This specifies the number of motifs used. Pass this argument only if `<FEATURE_TYPE>` is `m` or `b`. 

IMPORTANT:

Before using `<N_MOTIFS>` to test different numbers of motifs, it is necessary to first run training **without** the `<N_MOTIFS>` parameter. This initial run uses all motifs and computes SHAP value rankings, which are then saved as a CSV file. This file is required when you later pass the `<N_MOTIFS>` argument.

For example:
1. Run training without `<N_MOTIFS>`: `python train_model.py email-Enron xgb time b`
2. The ranking will be saved to `results_our_and_motifs/time/email-Enron/metrics/shap_ranking_xgb.csv`
3. Run training with `<N_MOTIFS>`: `python train_model.py email-Enron xgb time b 10`
4. Results will be saved to `results_our_and_motifs/time/email-Enron/metrics/test_xgb_10.csv`

# Results
Results obtained for the test set are saved in:
- Our model: `results_our/<SPLIT_TYPE>/<DATASET>/metrics/test_<MODEL_NAME>.csv`
- Our model + motifs: `results_our_and_motifs/<SPLIT_TYPE>/<DATASET>/metrics/test_<MODEL_NAME>.csv`
- Baseline model: `result_motifs/<SPLIT_TYPE>/<DATASET>/metrics/test_<MODEL_NAME>.csv`

# Additional scripts:
- `compute_features_update.py` used by `train_model.py` to compute our features
- `baseline\find_motifs.py` used by `\baseline\construct_motif_feature.py`
- `create_plots.ipynb` creates barplots with results






