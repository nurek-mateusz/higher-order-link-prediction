import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


datasets = ['coauth-MAG-Geology', 'coauth-MAG-History', 'contact-high-school', 'contact-primary-school', 'email-Enron', 'email-Eu', 
                    'NDC-classes', 'NDC-substances', 'threads-ask-ubuntu', 'tags-ask-ubuntu']

def save_trg(trg, y, path):
    a = np.array(trg, dtype=np.int32).reshape(-1, 3)
    d = {'v0': a[:, 0], 'v1': a[:, 1], 'v2': a[:, 2]}
    if y is not None:
        d['y'] = np.array(y, dtype=np.int8)
    pq.write_table(pa.table(d), path, compression='zstd')

def save_simplices(simplices, path):
    t, s = zip(*simplices) if simplices else ((), ())
    pq.write_table(pa.table({'time': np.array(t, dtype=np.int64), 
                             'nodes': pa.array([list(x) for x in s], type=pa.list_(pa.int32()))}),
                   path, compression='zstd')
    
def load_trg(path, with_y=False):
    tb = pq.read_table(path)
    trg = list(zip(tb['v0'].to_numpy(), tb['v1'].to_numpy(), tb['v2'].to_numpy()))
    if with_y:
        return trg, tb['y'].to_numpy()
    return trg

def load_simplices(path):
    tb = pq.read_table(path)
    return set(zip(tb['time'].to_pylist(), (tuple(x) for x in tb['nodes'].to_pylist())))