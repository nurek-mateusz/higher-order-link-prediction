import argparse
import gc
from itertools import combinations
import os
import pickle
import numpy as np
import pandas as pd


from sklearn.metrics import average_precision_score, roc_auc_score

OUTDIR = "results_baseline/time"

DATASETS =[
    "coauth-MAG-Geology",
    "coauth-MAG-History",
    "contact-high-school",
    "contact-primary-school-2",
    "email-Enron",
    "email-Eu",
    "NDC-classes",
    "NDC-substances",
    "tags-ask-ubuntu",
    "threads-ask-ubuntu"
]

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_simplices(dataset, split_type, data_type):
    path = os.path.join("processing_dataset", split_type, dataset)
    simplices_file = os.path.join(path, f"simplices_{data_type}.pickle")
    if not os.path.exists(simplices_file):
        return None
    with open(simplices_file, "rb") as f:
        simplices = pickle.load(f)
    return simplices

def count_timestamps(simplices):
    # (Previous) version: count unique timestamps in simplices
    if simplices is None:
        return None
    ts = set()
    for simplex in simplices:
        ts.add(simplex[0])  # timestamp is the first element of simplex
    return len(ts)

def count_timestamps_span(simplices):
    # (Current) version: count timestamps span in simplices, since some datasets have uneven timestamp distribution
    if simplices is None:
        return None
    min_ts = float("inf")
    max_ts = float("-inf")
    for simplex in simplices:
        ts = simplex[0]  # timestamp is the first element of simplex
        min_ts = min(min_ts, ts)
        max_ts = max(max_ts, ts)
    return max_ts - min_ts + 1

def _sample_bins(candidates, max_bins):
    # sample n(max_bins) bins from candidates, including the first and last one
    if max_bins is None or max_bins <= 0 or len(candidates) <= max_bins:
        return sorted(candidates)

    sampled_bins = {candidates[0], candidates[-1]}
    step = (len(candidates) - 1) / float(max_bins - 1)
    for i in range(max_bins - 2):
        idx = int(round((i + 1) * step))
        sampled_bins.add(candidates[idx])

    sampled_bins = sorted(sampled_bins)
    while len(sampled_bins) > max_bins:
        sampled_bins.pop(len(sampled_bins) // 2)
    return sampled_bins

def dynamic_bin_candidates(timespan, max_bins):
    """
    Bulid dynamic bin candidates based per-dataset timestamp timespan.
    Always include extreme bins:
    - bin=1 (only one bin for all timestamps)
    - bin=timespan (each timestamp is a separate bin)
    """
    if timespan is None or timespan <= 0:
        return [1]

    candidates = set()
    candidates.add(1)
    candidates.add(timespan)
    # add some fixed candidtes
    fix_bins = [2, 3, 5, 10, 20, 40, 80, 160]
    for b in fix_bins:
        if b < timespan:
            candidates.add(b)
    # add proportion-based cadidates
    proportions = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    for p in proportions:
        bin = int(timespan * p)
        if bin > 1 and bin < timespan:
            candidates.add(bin)
    # add power-of-2-based candidates
    p = 1
    while p < timespan:
        if p > 1 and p < timespan:
            candidates.add(p)
        p *= 2

    if max_bins is None or max_bins <= 0:
        return sorted(candidates)

    bin_candidates = _sample_bins(sorted(candidates), max_bins)
    return bin_candidates


def parse_bin_candidates(bins):
    if bins == "dynamic":
        return None
    else:
        # bins = "1,2,3" -> [1, 2, 3]
        return [int(b.strip()) for b in bins.split(",") if b.strip()]
    
def load_processing_pickle(dataset, split_type, data_type):
    # load trg_open, y, simplices from processing pickle files
    path = os.path.join("processing_dataset", split_type, dataset)
    trg_open_file = os.path.join(path, f"trg_open_{data_type}.pickle")
    y_file = os.path.join(path, f"y_{data_type}.pickle")
    simplices_file = os.path.join(path, f"simplices_{data_type}.pickle")

    with open(trg_open_file, "rb") as f:
        trg_open = pickle.load(f)
    with open(y_file, "rb") as f:
        y = pickle.load(f)
    with open(simplices_file, "rb") as f:
        simplices = pickle.load(f)

    return trg_open, y, simplices

def build_ts_to_bin(simplices, bin):
    if simplices is None:
        return {}

    ts = sorted({simplex[0] for simplex in simplices})
    if not ts:
        return {}

    min_ts = ts[0]
    max_ts = ts[-1]
    range_ts = max_ts - min_ts + 1

    if bin is None or bin <= 0:
        return {t: 0 for t in ts}  # all timestamps in one bin
    if bin >= range_ts:
        return {t: i for i, t in enumerate(ts)}  # each timestamp gets its own bin id

    # fixed-width binning
    # empty bins are allowed (timestamps may be sparse)
    ts_to_bin = {}
    for t in ts:
        b = int((t - min_ts) / float(range_ts) * bin)
        if b >= bin:  # edge case for max_ts
            b = bin - 1
        ts_to_bin[t] = b
    return ts_to_bin


def build_temporal_neighbor(simplices, bin):
    # Build temporal neighbors for each node based on simplices and timestamp bins.
    # Output:
    # - node_to_times: dict mapping node to set of time bins it appears in, e.g., {node1: {0, 1}, node2: {1}}
    # - node_time_to_neighbors: dict mapping (node, time_bin) to set of neighboring nodes in the same time bin, e.g., {(node1, 0): {node2, node3}, (node1, 1): {node4}, (node2, 1): {node1}}
    node_to_times = {}
    node_time_to_neighbors = {}
    # ts_to_bin is a dict mapping timestamp to bin id, e.g., {0: 0, 1: 0, 2: 1, 3: 1} for bin=2
    ts_to_bin = build_ts_to_bin(simplices, bin)
    for simplex in simplices:
        ts = int(simplex[0])  # timestamp is the first element of simplex
        nodes = simplex[1]  # the rest are nodes list in the simplex
        if len(nodes) < 2:
            continue  # skip simplices with less than 2 nodes
        t_bin = ts_to_bin.get(ts, 0) 
        for u in nodes:
            node_to_times.setdefault(u, set()).add(t_bin)
        for u,v in combinations(nodes, 2):
            node_time_to_neighbors.setdefault((u, t_bin), set()).add(v)
            node_time_to_neighbors.setdefault((v, t_bin), set()).add(u)
    return node_to_times, node_time_to_neighbors

def THCN(trg, node_to_times, node_time_to_neighbors):
    # temporal higher-order common neighbor (THCN)
    a,b,c = trg
    ta = node_to_times.get(a, set())
    tb = node_to_times.get(b, set())
    tc = node_to_times.get(c, set())
    t_all = ta|tb|tc

    CN_set = set()
    for t in t_all:
        na_t = node_time_to_neighbors.get((a, t), set())
        nb_t = node_time_to_neighbors.get((b, t), set())
        nc_t = node_time_to_neighbors.get((c, t), set())
        CN_set.update(na_t & nb_t & nc_t)
    return len(CN_set)


def run_hcn(trg_open, y, simplices, bin, max_samples):
    # calculate THCN with given timewindow bins
    node_to_times, node_time_to_neighbors = build_temporal_neighbor(simplices, bin)
    # max_samples is for faster evaluation during hyperparameter tuning, using a subset of data(for huge datasets)
    if max_samples is not None and len(trg_open) > max_samples:
        trg_iter = trg_open[:max_samples]
        y_iter = y[:max_samples]
    else:
        trg_iter = trg_open
        y_iter = y
    
    scores = [] # THCN prediction scores for each trg
    labels = [] # 0/1 labels for each trg
    for i,trg in enumerate(trg_iter):
        if i >= len(y_iter):
            break
        scores.append(THCN(trg, node_to_times, node_time_to_neighbors))
        labels.append(y_iter[i])
    return labels, scores

def evaluate_scores(y_true, y_scores):
    # calculate average precision and AUC, and performance
    try:
        ap = average_precision_score(y_true, y_scores)
    except Exception:
        ap = float("nan")

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = float("nan")
    
    random_baseline = sum(y_true) / len(y_true) if len(y_true) > 0 else 0
    performance = ap / random_baseline if random_baseline > 0 else float("nan")
    return {
        "average_precision": ap,
        "roc_auc": auc,
        "performance": performance
    }

def run_hcn_with_bins(dataset, split_type, bin_candidates, max_samples, top_k):
    # core function: find the best timewindow bin based on validation set(ensemble strategy)
    sweep_results = [] 
    val_results = []

    # using validation set for evaluation
    trg_open_val, y_val, simplices_val = load_processing_pickle(dataset, split_type, "val")

    for bin in bin_candidates:
        y_val_pred, s_val_pred = run_hcn(trg_open_val, y_val, simplices_val, bin, max_samples)
        val_metric = evaluate_scores(y_val_pred, s_val_pred)
        sweep_results.append({
            "dataset": dataset,
            "bin": bin,
            "val_performance": val_metric["performance"],
            "val_average_precision": val_metric["average_precision"],
            "val_roc_auc": val_metric["roc_auc"]
        })

        if not np.isnan(val_metric["performance"]):
            val_results.append((bin, val_metric["performance"] , val_metric["roc_auc"]))
        
    val_results.sort(key=lambda x: x[1], reverse=True)  # sort by performance

    del trg_open_val, y_val, simplices_val  # free memory
    gc.collect()

    best_bin = val_results[0][0] if val_results else None
    best_val_performance = val_results[0][1] if val_results else float('nan')
    best_val_auc = val_results[0][2] if val_results else float('nan')
    # ensemble top k timewindows for evaluation on test set
    top_k_results = val_results[:top_k] if val_results else []
    print(f"[{dataset}] Ensenbling Test Scores over Top-{top_k} bins: {[x[0] for x in top_k_results]}")

    # using test set for evaluation
    trg_open_test, y_test, simplices_test = load_processing_pickle(dataset, split_type, "test")

    ensemble_scores = None
    y_test_ref = None
    for bin, _, _ in top_k_results:
        y_test_pred, s_test_pred = run_hcn(trg_open_test, y_test, simplices_test, bin, max_samples)
        s_test_pred = np.asarray(s_test_pred, dtype=np.float32)
        if y_test_ref is None:
            y_test_ref = np.asarray(y_test_pred, dtype=np.int32)
            ensemble_scores = np.zeros_like(s_test_pred)
        ensemble_scores += s_test_pred
        del s_test_pred
        gc.collect()

    del trg_open_test, y_test, simplices_test
    gc.collect()

    if ensemble_scores is not None and len(top_k_results) > 0:
        ensemble_scores /= len(top_k_results)
        test_metric = evaluate_scores(y_test_ref, ensemble_scores)
    else:
        test_metric = {"average_precision": float("nan"), "roc_auc": float("nan"), "performance": float("nan")}

    best_results = {
        "dataset": dataset,
        "best_bin": best_bin,
        "top_k_bins": [x[0] for x in top_k_results], 
        "val_performance": best_val_performance,
        "val_auc": best_val_auc,
        "test_performance": test_metric["performance"],
        "test_auc": test_metric["roc_auc"]
    }
    return best_results, sweep_results

def main():
    """
    args: split_type, dataset, bin_candidates, max_samples, max_dynamic_bins, top_k
    default values: 
    split_type="time", 
    dataset=all datasets, 
    bin_candidates="dynamic", 
    max_samples=None, 
    max_dynamic_bins=20, 
    top_k=3
    """
    parser = argparse.ArgumentParser(description="find best timewindow for HCN")
    parser.add_argument("--split_type", type=str, default="time", help="type of split")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=",".join(DATASETS), 
        help="dataset names"
        )
    parser.add_argument(
        "--bin_candidates",
        type=str,
        default="dynamic",
        help="candidates of timewindow bins, separated by comma, e.g., '1,2,3' or 'dynamic'",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="optional--maximum number of samples to use for evaluation (for faster evaluation)",
    )
    parser.add_argument(
        "--max_dynamic_bins",
        type=int,
        default=20,
        help="number of bins for dynamic timewindow",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="esenmble top k timewindows for evaluation"
    )
    args = parser.parse_args()

    # timestamp window candidates preparation
    datasets = args.dataset.split(",")
    static_bins = parse_bin_candidates(args.bin_candidates)

    all_best_results = []
    for dataset in datasets:
        print(f"======{dataset}======")
        train_simplices = load_simplices(dataset, args.split_type, "train")
        # two version for timestamp bin candidates:
        # 1. count unique timestamps in simplices (previous version)
        # 2. count timestamps span (current version) -- uneven timestamp distribution for some datasets.
        timespan = count_timestamps_span(train_simplices)
        if static_bins is None:
            bin_candidates = dynamic_bin_candidates(timespan, args.max_dynamic_bins)
        else:
            bin_candidates = static_bins
        

        best_results, sweep_results = run_hcn_with_bins(
            dataset, 
            args.split_type, 
            bin_candidates, 
            args.max_samples, 
            args.top_k
        )
        """
        -- best_results(dict):
            dataset, best_bin, top_k_bins, val_performance, val_auc, test_performance, test_auc
        -- sweep_results(list of dict):
            dataset, bin, val_performance, val_average_precision, val_roc_auc
        """
        best_results['n_timespan_train'] = timespan
        # save sweep_results to csv for each dataset
        dataset_dir = os.path.join(OUTDIR, dataset)
        ensure_dir(dataset_dir)
        sweep_df = pd.DataFrame(sweep_results)
        sweep_df.to_csv(os.path.join(dataset_dir, f"{dataset}_sweep_results_THCN.csv"), index=False)
        print(f"save sweep results to {os.path.join(dataset_dir, f'{dataset}_sweep_results_THCN.csv')}")

        all_best_results.append(best_results)

    # save summary results to csv
    ensure_dir(OUTDIR)
    summary_file = os.path.join(OUTDIR, "summary_best_results_THCN.csv")
    df_all_best_results = pd.DataFrame(all_best_results)

    if os.path.exists(summary_file):
        df_all_best_results.to_csv(summary_file, index=False, header=False, mode='a')
    else:
        df_all_best_results.to_csv(summary_file, index=False)

    print(f"save summary results to {summary_file}")



if __name__ == "__main__":
    main()