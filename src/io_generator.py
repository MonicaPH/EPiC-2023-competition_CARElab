import pandas as pd
import numpy as np
from pathlib import Path
import os, sys, multiprocessing, re
from functools import partial
from utils import check_dir

import logging, datetime

import warnings
warnings.filterwarnings("ignore")

log_format = '%(asctime)s [%(levelname)s] %(message)s'
log_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(format=log_format, 
                    force=True,
                    handlers=[
                        logging.FileHandler(f"../log/{log_filename}.log"),
                        logging.StreamHandler()
                        ],
                    level=logging.INFO
                    )

def create_sliding_window(X, y, past_window_size=50, future_window_size=50):
    """
        X: DataFrame
        window_size
    """
    new_cols_past = None
    new_cols_future = None
    if past_window_size > 0:
        new_cols_past = pd.concat([X.shift(i) for i in range(past_window_size+1)], axis=1)
    if future_window_size > 0:
        new_cols_future = pd.concat([X.shift(-i) for i in range(1, future_window_size+1)], axis=1)
    
    new_X = pd.concat([new_cols_past, new_cols_future], axis=1).dropna()
    
    start = np.intersect1d(y.index.tolist(), new_X.index.tolist()).min()
    end = np.intersect1d(y.index.tolist(), new_X.index.tolist()).max()

    return new_X.loc[y.loc[start:end+1].index.tolist()], y.loc[start:end+1]

def load_data_dict(prefix='../'):
    scenarios = [1, 2, 3, 4]
    folds = [[-1], [0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1]]
    
    data_dict = {}
    data_dict['scenarios'] = scenarios
    data_dict['folds'] = folds
    for i, scenario in enumerate(scenarios):
        data_dict[scenario] = dict()
        
        for fold in folds[i]:
            data_dict[scenario][fold] = dict()
            
            filenames = os.listdir(Path(prefix) / 
                                   f'data/scenario_{scenario}' /
                                   f'{"fold_" + str(fold) if fold != -1 else ""}' / 'train/physiology')
            train_subs = sorted(list(set([int(re.findall(r'(?<=sub_)\d+', s)[0]) for s in filenames])))
            train_vids = sorted(list(set([int(re.findall(r'(?<=vid_)\d+', s)[0]) for s in filenames])))

            filenames = os.listdir(Path(prefix) / 
                                   f'data/scenario_{scenario}' /
                                   f'{"fold_" + str(fold) if fold != -1 else ""}' / 'test/physiology')
            test_subs = sorted(list(set([int(re.findall(r'(?<=sub_)\d+', s)[0]) for s in filenames])))
            test_vids = sorted(list(set([int(re.findall(r'(?<=vid_)\d+', s)[0]) for s in filenames])))
            
            data_dict[scenario][fold]['train_subs'] = train_subs
            data_dict[scenario][fold]['train_vids'] = train_vids
            data_dict[scenario][fold]['test_subs'] = test_subs
            data_dict[scenario][fold]['test_vids'] = test_vids
    
    return data_dict

def load_model_dict(data_dict):
    model_dict = dict()
    # scenario 1, 240 models
    for scenario in data_dict['scenarios']:
        model_dict[scenario] = dict()
        
    for fold in data_dict[1].keys():
        model_dict[1][fold] = [[(sub, vid)] for sub in data_dict[1][-1]['train_subs'] for vid in data_dict[1][-1]['train_vids']]
    
    # scenario 2, 30 models
    for fold in data_dict[2].keys():
        model_dict[2][fold] = [[(sub, vid) for vid in data_dict[2][fold]['train_vids']] for sub in data_dict[2][fold]['train_subs']]
        
    # scenario 3, 2 models
    pass

    # scenario 4
    for fold in data_dict[4].keys():
        model_dict[4][fold] = [[(sub, vid) for sub in data_dict[4][fold]['train_subs']] for vid in data_dict[4][fold]['train_vids']]
       
    return model_dict

def load_raw_data(scenario, fold, sub, vid, train_test, prefix='../'):
    return pd.read_csv(Path(prefix) / 'data' /
                   f'scenario_{scenario}' /
                   f'{"fold_" + str(fold) if fold != -1 else ""}' / 
                   train_test / 'physiology' / f'sub_{sub}_vid_{vid}.csv', index_col='time')

def load_annotation(scenario, fold, sub, vid, train_test, prefix='../'):
    return pd.read_csv(Path(prefix) / 'data' /
                   f'scenario_{scenario}' /
                   f'{"fold_" + str(fold) if fold != -1 else ""}' / 
                   train_test / 'annotations' / f'sub_{sub}_vid_{vid}.csv', index_col='time')

def load_clean_data(scenario, fold, sub, vid, train_test, prefix='../'):
    cols = ['ECG_Clean', 'PPG_Clean', 'EDA_Clean', 'RSP_Clean', 'EMG_Clean_zygo', 'EMG_Clean_coru', 'EMG_Clean_trap']
    return pd.read_csv(Path(prefix) / 'clean_data' /
                       f'scenario_{scenario}' /
                       f'{"fold_" + str(fold) if fold != -1 else ""}' / 
                       train_test / f'sub_{sub}_vid_{vid}.csv', index_col='time')[cols]

def load_features(scenario, fold, sub, vid, train_test, prefix='../'):
    return pd.read_csv(Path(prefix) / f'features/scenario_{scenario}/' /
                f'{"fold_" + str(fold) if fold != -1 else ""}' / 
                train_test / f'sub_{sub}_vid_{vid}.csv', index_col='time').fillna(0)

def load_io(scenario, fold, sub, vid, train_test, prefix, past_window_size, future_window_size):
    y = load_annotation(scenario, fold, sub, vid, train_test, prefix)
    X = load_clean_data(scenario, fold, sub, vid, train_test, prefix)
    X, y = create_sliding_window(X, y, past_window_size, future_window_size)
    features = load_features(scenario, fold, sub, vid, train_test, prefix).loc[y.index]
    X = pd.concat([X, features], axis=1)

    return X, y

def func(sub_vid_pairs, scenario, fold, prefix, input_path, output_path, past_window_size, future_window_size):
    Xs = None
    ys = None

    s, v = 0, 0

    for sub, vid in sub_vid_pairs:
        X, y = load_io(scenario, fold, sub, vid, 'train', prefix, past_window_size, future_window_size)
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

def save_training_io(scenario, past_window_size=50, future_window_size=50, prefix='../'):
    data_dict = load_data_dict()
    model_dict = load_model_dict(data_dict)
    
    for fold, model_list in model_dict[scenario].items():
        input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'train' / 'physiology'
        output_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'train' / 'annotations'
        check_dir(input_path, output_path)

        specific_func = partial(func, scenario=scenario, fold=fold,
                                prefix=prefix, input_path=input_path, output_path=output_path,
                                past_window_size=past_window_size, future_window_size=future_window_size)

        pool_obj = multiprocessing.Pool()
        pool_obj.map(specific_func, model_list)
        pool_obj.close()

def save_test_io(scenario, past_window_size=50, future_window_size=50, prefix='../'):
    data_dict = load_data_dict()
    for fold, model_list in model_dict[scenario].items():
        input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'train' / 'physiology'
        output_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'train' / 'annotations'
        check_dir(input_path, output_path)

        specific_func = partial(func, scenario=scenario, fold=fold,
                                prefix=prefix, input_path=input_path, output_path=output_path,
                                past_window_size=past_window_size, future_window_size=future_window_size)

        pool_obj = multiprocessing.Pool()
        pool_obj.map(specific_func, model_list)
        pool_obj.close()

