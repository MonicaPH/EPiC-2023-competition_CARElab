import pandas as pd
import multiprocessing
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor

import sys, os
sys.path.append(os.path.relpath("../src/"))
from utils import check_dir
import warnings
import logging, datetime

# warnings.filterwarnings("ignore")

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

def train(X, y, model_path, num_cpus, num_gpus):
    train_data = pd.concat([X, y.drop(['valence'], axis=1)], axis=1)
    train_data = TabularDataset(train_data)
    predictor = TabularPredictor(label='arousal', problem_type='regression', path=str(model_path) + '_arousal', verbosity=0).fit(train_data, ag_args_fit={'num_cpus': num_cpus, 'num_gpus': num_gpus})
    logging.info('arousal model fitted.')

    train_data = pd.concat([X, y.drop(['arousal'], axis=1)], axis=1)
    train_data = TabularDataset(train_data)
    predictor = TabularPredictor(label='valence', problem_type='regression', path=str(model_path) + '_valence', verbosity=0).fit(train_data, ag_args_fit={'num_cpus': num_cpus, 'num_gpus': num_gpus})
    logging.info('valence model fitted.')

prefix = '../'

# subs: [0, 1, 2, 3, 5,     hpc
#        7, 9, 11, 12,      xin
#        13, 14, 15, 17, 18,kun
#        19, 20, 23, 24,    rtx titan
#        26, 27, 28, 29,    1080ti1
#        30, 31, 32, 34,    2080ti
#        41, 43, 44, 45]    1080ti
#
scenario = 2
subs = [30, 31, 32, 34]
num_gpus = 2

input_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'train' / 'physiology'
output_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'train' / 'annotations'
model_path = Path(prefix) / f'models/scenario_{scenario}'
check_dir(model_path)

for sub in subs:
    logging.info(f'start sub {sub} ...')
    X = pd.read_csv(input_path / f'sub_{sub}.csv', index_col='time')
    y = pd.read_csv(output_path / f'sub_{sub}.csv', index_col='time')
    train(X, y, model_path / f'sub_{sub}', multiprocessing.cpu_count(), num_gpus)
    logging.info(f'finish sub {sub}.')
