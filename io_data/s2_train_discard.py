import os, sys
import multiprocessing
import pandas as pd
from functools import partial
from pathlib import Path
sys.path.append(os.path.relpath("../src/"))
from io_generator import load_data_dict, load_io
from utils import check_dir

model_list = []
sub_list = []
past_window_size = 50
future_window_size = 50

data_dict = load_data_dict()
for fold in data_dict['folds'][1]:
    for sub in data_dict[2][fold]['train_subs']:
        if sub not in sub_list:
            sub_list.append(sub)
            model_list.append([(fold, sub, vid) for vid in data_dict[2][fold]['train_vids']])
            
prefix = '../'
input_path = Path(prefix) / f'io_data/scenario_2/train/physiology'
output_path = Path(prefix) / f'io_data/scenario_2/train/annotations'
check_dir(input_path, output_path)

def func_scenario2(fold_sub_vid_pairs, input_path, output_path, past_window_size, future_window_size):
    Xs = None
    ys = None
    s = 0
    for fold, sub, vid in fold_sub_vid_pairs:
        s = sub
        X, y = load_io(2, fold, sub, vid, 'train', '../', past_window_size=50, future_window_size=50)
        Xs = pd.concat([Xs, X], axis=0)
        ys = pd.concat([ys, y], axis=0)

    Xs.to_csv(input_path / f'sub_{s}.csv', index_label='time')
    ys.to_csv(output_path / f'sub_{s}.csv', index_label='time')
    
specific_func = partial(func_scenario2, 
                        input_path=input_path,
                        output_path=output_path,
                        past_window_size=past_window_size,
                        future_window_size=future_window_size)

pool_obj = multiprocessing.Pool()
pool_obj.map(specific_func, model_list)
pool_obj.close()
