# higher-order-link-prediction

# How to Run
1. Manually copy MAG dataset to repo (too large to commit on github) 
2. Run `preprocessing.ipynb` (section *Prepare data for ScHoLP library*).
   Created files must be saved in `scholp/data/DATASET_NAME/` as:
   - `DATASET_NAME-nverts.txt`,
   - `DATASET_NAME-simplices.txt`,
   - `DATASET_NAME-times.txt`.
4. Motif analysis code is in `/scholp/run.ipynb` ([Julia](https://julialang.org/downloads/) is required)

