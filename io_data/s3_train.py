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

prefix = '../'
input_path = Path(prefix) / f'io_data/scenario_3/train/physiology'
output_path = Path(prefix) / f'io_data/scenario_3/train/annotations'
check_dir(input_path, output_path)

data_dict = load_data_dict()
subs = [0, 3, 4, 5, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21,
        22, 23, 26, 30, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44]

# fold 0
vids = [0, 3, 4, 21]
fold = 0
arousal_1_list = [(fold, sub, vid) for sub in subs for vid in vids]

Xs = None
ys = None
for fold, sub, vid in arousal_1_list:
    X, y = load_io(3, fold, sub, vid, 'train', '../', past_window_size=50, future_window_size=50)
    Xs = pd.concat([Xs, X], axis=0)
    ys = pd.concat([ys, y], axis=0)

Xs.to_csv(input_path / f'arousal_03421.csv', index_label='time')
ys.to_csv(output_path / f'arousal_03421.csv', index_label='time')

print('arousal_03421 saved.')

vids = [4, 10, 21, 22]
fold = 0
valence_1_list = [(fold, sub, vid) for sub in subs for vid in vids]

Xs = None
ys = None
for fold, sub, vid in valence_1_list:
    X, y = load_io(3, fold, sub, vid, 'train', '../', past_window_size=50, future_window_size=50)
    Xs = pd.concat([Xs, X], axis=0)
    ys = pd.concat([ys, y], axis=0)

Xs.to_csv(input_path / f'valence_4102122.csv', index_label='time')
ys.to_csv(output_path / f'valence_4102122.csv', index_label='time')

print('valence_4102122 saved.')

# fold 3
vids = [10, 16, 20, 22]
fold = 3
arousal_2_list = [(fold, sub, vid) for sub in subs for vid in vids]

Xs = None
ys = None
for fold, sub, vid in arousal_2_list:
    X, y = load_io(3, fold, sub, vid, 'train', '../', past_window_size=50, future_window_size=50)
    Xs = pd.concat([Xs, X], axis=0)
    ys = pd.concat([ys, y], axis=0)

Xs.to_csv(input_path / f'arousal_10162022.csv', index_label='time')
ys.to_csv(output_path / f'arousal_10162022.csv', index_label='time')

vids = [0, 3, 16, 20]
fold = 3
valence_2_list = [(fold, sub, vid) for sub in subs for vid in vids]

Xs = None
ys = None
for fold, sub, vid in valence_2_list:
    X, y = load_io(3, fold, sub, vid, 'train', '../', past_window_size=50, future_window_size=50)
    Xs = pd.concat([Xs, X], axis=0)
    ys = pd.concat([ys, y], axis=0)

Xs.to_csv(input_path / f'valence_031620.csv', index_label='time')
ys.to_csv(output_path / f'valence_031620.csv', index_label='time')
