import os, sys
import numpy as np
import multiprocessing
import pandas as pd
from functools import partial
from pathlib import Path
sys.path.append(os.path.relpath("../src/"))
from io_generator import load_data_dict, load_io
from utils import check_dir


scenario = 4
prefix = '../'
past_window_size = 50
future_window_size = 50

def func(fold_sub_vid, scenario, prefix, past_window_size, future_window_size):
    fold, sub, vid = fold_sub_vid
    X, y = load_io(scenario, fold, sub, vid, 'test', prefix, 
                   past_window_size=past_window_size, future_window_size=future_window_size)
    X.to_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
    y.to_csv(output_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')

folds_subs_vids = []
data_dict = load_data_dict()
for fold in data_dict[scenario].keys():
    input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test' / 'physiology'
    output_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test' / 'annotations'
    check_dir(input_path, output_path)
    test_subs, test_vids = data_dict[scenario][fold]['test_subs'], data_dict[scenario][fold]['test_vids']
    folds_subs_vids.append([(fold, sub, vid) for sub in test_subs for vid in test_vids])
folds_subs_vids = np.array(folds_subs_vids).reshape(-1, 3)

specific_func = partial(func, 
                        scenario=scenario,
                        prefix=prefix,
                        past_window_size=past_window_size,
                        future_window_size=future_window_size)

pool_obj = multiprocessing.Pool()
pool_obj.map(specific_func, folds_subs_vids)
pool_obj.close()

