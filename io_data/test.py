import os, sys
import numpy as np
import multiprocessing
import pandas as pd
from functools import partial
from pathlib import Path
sys.path.append(os.path.relpath("../src/"))
from dataloader import S1, S2, S3, S4
from io_generator import load_data_dict, load_io
from utils import check_dir
import argparse

parser = argparse.ArgumentParser(description='generate input of test data')
parser.add_argument('-s', '--scenario', type=int, required=True)
parser.add_argument('-f', '--fold', type=int, action='append')
parser.add_argument('--past_window_size', type=int, default=50)
parser.add_argument('--future_window_size', type=int, default=50)
args = parser.parse_args()

scenario = args.scenario

assert scenario in [1, 2, 3, 4], 'scenario: 1, 2, 3, 4'

if scenario == 1:
    s = S1()
elif scenario == 2:
    s = S2()
elif scenario == 3:
    s = S3()
else:
    s = S4()

folds = args.fold if args.fold is not None else s.fold
past_window_size = args.past_window_size
future_window_size = args.future_window_size
#
def func(fold_sub_vid, scenario, prefix, past_window_size, future_window_size):
    fold, sub, vid = fold_sub_vid
    input_path = Path(f'io_data/scenario_{scenario}') / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test' / 'physiology'
    output_path = Path(f'io_data/scenario_{scenario}') / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test' / 'annotations'
    X, y = load_io(scenario, fold, sub, vid, 'test', prefix, 
                   past_window_size=past_window_size, future_window_size=future_window_size)
    X.to_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
    y.to_csv(output_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
#
# folds_subs_vids = []
# data_dict = load_data_dict()
# for fold in data_dict[scenario].keys():
#     input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test' / 'physiology'
#     output_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test' / 'annotations'
#     check_dir(input_path, output_path)
#     test_subs, test_vids = data_dict[scenario][fold]['test_subs'], data_dict[scenario][fold]['test_vids']
#     folds_subs_vids.append([(fold, sub, vid) for sub in test_subs for vid in test_vids])
# folds_subs_vids = np.array(folds_subs_vids).reshape(-1, 3)
#
# specific_func = partial(func, 
#                         scenario=scenario,
#                         prefix=prefix,
#                         past_window_size=past_window_size,
#                         future_window_size=future_window_size)
#
# pool_obj = multiprocessing.Pool()
# pool_obj.map(specific_func, folds_subs_vids)
# pool_obj.close()
#
