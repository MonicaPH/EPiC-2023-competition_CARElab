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

scenario = 1
subs = [20, 22, 26, 28, 29]
vids = [1, 9, 10, 11, 13, 14, 18, 20]
num_gpus = 2

input_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'train' / 'physiology'
output_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'train' / 'annotations'
model_path = Path(prefix) / f'models/scenario_{scenario}'
check_dir(model_path)

for sub in subs:
    for vid in vids:
        logging.info(f'start sub {sub} vid {vid}...')
        X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
        y = pd.read_csv(output_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
        train(X, y, model_path / f'sub_{sub}_vid_{vid}', multiprocessing.cpu_count(), num_gpus)
        logging.info(f'finish sub {sub} vid {vid}.')
