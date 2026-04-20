# This is modified version of file read_simplices from the repository https://github.com/Rm-Y/Simplicial-Motif-Predictor-Method
import pandas as pd
import os
import sys
import pickle
import networkx as nx
import numpy as np
import itertools

np.random.seed(0)

def split_file_to_list(fl_path):
    """Reads the data from a file and returns a list."""
    a = []
    with open(fl_path, 'r') as f:
        for ff in f.readlines():
            a.append(int(ff))
    return a

def find_tri_common_neighbor(G):
    """Finds and returns all triangles (2-clique) in the graph G."""
    neighbor_dict = {n: set(G.neighbors(n)) for n in G.nodes()}
    all_triangle = set()
    for node1, node2 in G.edges():
        common_neighbors = neighbor_dict[node1].intersection(neighbor_dict[node2])
        for common_node in common_neighbors:
            triangle = tuple(sorted([node1, common_node, node2]))
            all_triangle.add(triangle)
    return all_triangle

def split_train_probe(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio, k, dataset):
    """Splits the simplicial data into open/closed triangles, corresponding skeleton networks, and dataset labels based on time. 
    start_ratio represents the starting timestamp for dataset splitting, 
    end_ratio represents the ending timestamp for dataset splitting, 
    test_ratio represents the ending timestamp for the test set, 
    k represents the order of the prediction, and 2 represents three nodes."""

    if dataset == 'contact-high-school':
        simplices_0_60, simplices_60_80, train_zip = split_high_school_time(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio)
    elif dataset == 'contact-primary-school':
        simplices_0_60, simplices_60_80, train_zip = split_primary_school_time(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio)
    else:
        simplices_0_60, simplices_60_80, train_zip = split_data_time(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio)
    
    train_closed_trg, train_edge = set(), set()
    for simp in simplices_0_60:
        if len(simp) == 2:
            train_edge.add(tuple(sorted(simp)))
        elif len(simp) > 2:
            for trig in itertools.combinations(simp, k):
                train_closed_trg.add(tuple(sorted(trig)))
            for edg in itertools.combinations(simp, 2):
                train_edge.add(tuple(sorted(edg)))
    
    G_train = nx.Graph()
    G_train.add_edges_from(train_edge)
    train_node = set(G_train.nodes())
    all_triangle_train = find_tri_common_neighbor(G_train)
    train_open_trg = list(all_triangle_train ^ train_closed_trg)

    closed_trg_60_80 = set()
    for simp in simplices_60_80:
        if len(simp) > 2:
            for trig in itertools.combinations(simp, k):
                if tuple(sorted(trig)) not in train_closed_trg:
                    if len(set(trig) & train_node) == k:
                        closed_trg_60_80.add(tuple(sorted(trig)))

    y_train = [1 if i in closed_trg_60_80 else 0 for i in train_open_trg]

    return G_train, train_open_trg, list(train_closed_trg), y_train, train_zip

def split_data(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio):
    """Splits data into training and testing sets based on the timestamps."""
    old_zip, new_zip = set(), set()
    old_simplices, new_simplices = [], []
    
    # Split data based on time
    start_time = int(np.round(np.percentile(tm_lis, start_ratio)))
    end_time = int(np.round(np.percentile(tm_lis, min(end_ratio, 100))))

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if (time >= start_time) & (time <= end_time):
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            old_zip.add(simp_zip)
        curr_ind += nv
    if old_zip:
        _, old_simplices = zip(*old_zip)

    # Split testing set based on time
    start_time = end_time + 1
    end_time = int(np.round(np.percentile(tm_lis, min(end_ratio + test_ratio, 100))))

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if (time >= start_time) & (time <= end_time):
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            new_zip.add(simp_zip)

        curr_ind += nv
    if new_zip:
        _, new_simplices = zip(*new_zip)

    return old_simplices, new_simplices, old_zip

def split_data_time(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio):
    """Splits data into training and testing sets based on the timestamps."""
    old_zip, new_zip = set(), set()
    old_simplices, new_simplices = [], []
    
    # Split data based on time
    min_time = np.min(tm_lis)
    max_time = np.max(tm_lis)
    start_time = min_time
    end_time = start_time + (max_time - min_time) * (end_ratio/100)

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if (time >= start_time) & (time <= end_time):
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            old_zip.add(simp_zip)
        curr_ind += nv
    if old_zip:
        _, old_simplices = zip(*old_zip)

    # Split testing set based on time
    test_time = start_time + (max_time - min_time) * ((end_ratio + test_ratio)/100)

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if (time > end_time) & (time <= test_time):
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            new_zip.add(simp_zip)

        curr_ind += nv
    if new_zip:
        _, new_simplices = zip(*new_zip)

    return old_simplices, new_simplices, old_zip

def split_high_school_time(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio):
    """Splits data into training and testing sets based on the timestamps."""
    # Timestamps of communication gaps between five days of interaction
    #                   interval_1                  interval_2                  interval_3                  interval_4
    # high-school       (1385999980, 1386054020)    (1386086380, 1386140420)    (1386172780, 1386226820)    (1386259180, 1386313220)
    old_zip, new_zip = set(), set()
    old_simplices, new_simplices = [], []
    
    # Split data based on time
    if end_ratio == 70: # training set
        end_time = 1386140420
        test_time = 1386226820
    elif end_ratio == 80: # validation set
        end_time = 1386226820
        test_time = 1386313220
    else: # test set
        end_time = 1386313220
        test_time = np.max(tm_lis)

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if time <= end_time:
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            old_zip.add(simp_zip)
        curr_ind += nv
    if old_zip:
        _, old_simplices = zip(*old_zip)

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if (time > end_time) & (time <= test_time):
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            new_zip.add(simp_zip)

        curr_ind += nv
    if new_zip:
        _, new_simplices = zip(*new_zip)

    return old_simplices, new_simplices, old_zip

def split_primary_school_time(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio):
    """Splits data into training and testing sets based on the timestamps."""
    # Timestamps of communication gaps between two days of interaction
    #                   interval_1
    # primary-school    (62300, 117240)
    old_zip, new_zip = set(), set()
    old_simplices, new_simplices = [], []
    
    # Split data based on time
    if end_ratio == 70: # training set
        end_time = 117240
        test_time = 117240 + 0.5 * (np.max(tm_lis) - 117240)
    elif end_ratio == 80: # validation set
        end_time = 117240 + 0.25 * (np.max(tm_lis) - 117240)
        test_time = 117240 + 0.5 * (np.max(tm_lis) - 117240)
    else: # test set
        end_time = 117240 + 0.5 * (np.max(tm_lis) - 117240)
        test_time = np.max(tm_lis)

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if time <= end_time:
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            old_zip.add(simp_zip)
        curr_ind += nv
    if old_zip:
        _, old_simplices = zip(*old_zip)

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if (time > end_time) & (time <= test_time):
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            new_zip.add(simp_zip)

        curr_ind += nv
    if new_zip:
        _, new_simplices = zip(*new_zip)

    return old_simplices, new_simplices, old_zip

if __name__ == "__main__":    
    dataset_list = ['coauth-MAG-Geology', 'coauth-MAG-History', 'contact-high-school', 'contact-primary-school', 'email-Enron', 'email-Eu', 
                    'NDC-classes', 'NDC-substances', 'threads-ask-ubuntu', 'tags-ask-ubuntu']

    for dataset in dataset_list:
        print(dataset)
        fl_nm_nv = '../data/' + dataset + '/' + dataset + '-nverts.txt'
        fl_nm_sp = '../data/' + dataset + '/' + dataset + '-simplices.txt'
        fl_nm_tm = '../data/' + dataset + '/' + dataset + '-times.txt'

        nv_lis = split_file_to_list(fl_nm_nv)  # the number of vertices within each simplex
        sp_lis = split_file_to_list(fl_nm_sp)  # the nodes comprising the simplices
        tm_lis = split_file_to_list(fl_nm_tm)  # the timestamps for each simplex
    
        G_train, x_train_trg, train_closed, y_train, simplices_train = split_train_probe(nv_lis, sp_lis, tm_lis, 0, 70, 20, 3, dataset)
        G_val, x_val_trg, val_closed, y_val, simplices_val = split_train_probe(nv_lis, sp_lis, tm_lis, 0, 80, 10, 3, dataset)
        G_test, x_test_trg, test_closed, y_test, simplices_test = split_train_probe(nv_lis, sp_lis, tm_lis, 0, 90, 10, 3, dataset)
        
        # Print statistics of training and testing data
        print('Training set sample count:', len(x_train_trg))
        print('Validation set sample count:', len(x_val_trg))
        print('Testing set sample count:', len(x_test_trg))
        print('Training set closed triangles count:', sum(y_train))
        print('Training set open triangles count:', len(y_train) - sum(y_train))
        print('Validation set closed triangles count:', sum(y_val))
        print('Validation set open triangles count:', len(y_val) - sum(y_val))
        print('Testing set closed triangles count:', sum(y_test))
        print('Testing set open triangles count:', len(y_test) - sum(y_test))
        print('--------')

        # Save processed dataset
        save_dir = '../data/processed/' + dataset
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        with open(save_dir + '/trg_open_train.pickle', 'wb') as f:
            pickle.dump(x_train_trg, f)
        with open(save_dir + '/trg_closed_train.pickle', 'wb') as f:
            pickle.dump(train_closed, f)
        with open(save_dir + '/y_train.pickle', 'wb') as f:
            pickle.dump(y_train, f)
        with open(save_dir + '/G_train.pickle', 'wb') as f:
            pickle.dump(G_train, f)
        with open(save_dir + '/simplices_train.pickle', 'wb') as f:
            pickle.dump(simplices_train, f)

        with open(save_dir + '/trg_open_val.pickle', 'wb') as f:
            pickle.dump(x_val_trg, f)
        with open(save_dir + '/trg_closed_val.pickle', 'wb') as f:
            pickle.dump(val_closed, f)
        with open(save_dir + '/y_val.pickle', 'wb') as f:
            pickle.dump(y_val, f)
        with open(save_dir + '/G_val.pickle', 'wb') as f:
            pickle.dump(G_val, f)
        with open(save_dir + '/simplices_val.pickle', 'wb') as f:
            pickle.dump(simplices_val, f)

        with open(save_dir + '/trg_open_test.pickle', 'wb') as f:
            pickle.dump(x_test_trg, f)
        with open(save_dir + '/trg_closed_test.pickle', 'wb') as f:
            pickle.dump(test_closed, f)
        with open(save_dir + '/y_test.pickle', 'wb') as f:
            pickle.dump(y_test, f)
        with open(save_dir + '/G_test.pickle', 'wb') as f:
            pickle.dump(G_test, f)
        with open(save_dir + '/simplices_test.pickle', 'wb') as f:
            pickle.dump(simplices_test, f)