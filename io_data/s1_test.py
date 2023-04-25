import os, sys
sys.path.append(os.path.relpath("../src/"))
from io_generator import load_io, load_data_dict, load_model_dict
from utils import check_dir
from functools import partial
import multiprocessing
from pathlib import Path
import pandas as pd

def func(sub_vid_pairs, scenario, fold, prefix, input_path, output_path, past_window_size, future_window_size):
    Xs = None
    ys = None

    s, v = 0, 0

    for sub, vid in sub_vid_pairs:
        X, y = load_io(scenario, fold, sub, vid, 'test', prefix, past_window_size, future_window_size)
        Xs = pd.concat([Xs, X], axis=0)
        ys = pd.concat([ys, y], axis=0)
        s, v = sub, vid
    
    if scenario == 1:
        Xs.to_csv(input_path / f'sub_{s}_vid_{v}.csv', index_label='time')
        ys.to_csv(output_path / f'sub_{s}_vid_{v}.csv', index_label='time')
    elif scenario == 2:
        Xs.to_csv(input_path / f'sub_{s}.csv', index_label='time')
        ys.to_csv(output_path / f'sub_{s}.csv', index_label='time')
    elif scenario == 4:
        Xs.to_csv(input_path / f'vid_{v}.csv', index_label='time')
        ys.to_csv(output_path / f'vid_{v}.csv', index_label='time')

scenario = 1
past_window_size = 50
future_window_size = 50
prefix = '../'
data_dict = load_data_dict()
model_dict = load_model_dict(data_dict)

for fold, model_list in model_dict[scenario].items():
    input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test' / 'physiology'
    output_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test' / 'annotations'
    check_dir(input_path, output_path)

    specific_func = partial(func, scenario=scenario, fold=fold,
                            prefix=prefix, input_path=input_path, output_path=output_path,
                            past_window_size=past_window_size, future_window_size=future_window_size)

    pool_obj = multiprocessing.Pool()
    pool_obj.map(specific_func, model_list)
    pool_obj.close()

