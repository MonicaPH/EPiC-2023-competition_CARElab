import pandas as pd
import multiprocessing
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
from src.io_generator import load_data_dict, load_model_dict
import argparse

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--num_gpus', type=int, help='1', required=True, default=1)

args = parser.parse_args()

import warnings
import logging, datetime

warnings.filterwarnings("ignore")

log_format = '%(asctime)s [%(levelname)s] %(message)s'
log_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(format=log_format, 
                    force=True,
                    handlers=[
                        logging.FileHandler(f"log/{log_filename}.log"),
                        logging.StreamHandler()
                        ],
                    level=logging.INFO
                    )

def train(X, y, model_path, num_cpus, num_gpus):
    train_data = pd.concat([X, y.drop('valence')], axis=1)
    train_data = TabularDataset(train_data)
    predictor = TabularPredictor(label='arousal', problem_type='regression', path=model_path / '_arousal', verbosity=0).fit(train_data, ag_args_fit={'num_cpus': num_cpus, 'num_gpus': num_gpus})
    logging.info('arousal model fitted.')

    train_data = pd.concat([X, y.drop('arousal')], axis=1)
    train_data = TabularDataset(train_data)
    predictor = TabularPredictor(label='valence', problem_type='regression', path=model_path / '_valence', verbosity=0).fit(train_data, ag_args_fit={'num_cpus': num_cpus, 'num_gpus': num_gpus})
    logging.info('valence model fitted.')

prefix = './'

data_dict = load_data_dict()
model_dict = load_model_dict()

scenario = 1
for fold, model_list in model_dict[scenario].items():
    input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'train' / 'physiology'
    output_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}' / 'train' / 'annotations'
    model_path = Path(prefix) / f'models/scenario_{scenario}' / f'{"fold_" + str(fold) if fold != -1 else ""}'

    for sub_vid_pairs in model_list:
        for sub, vid in sub_vid_pairs:
            logging.info(f'start sub {sub} vid {vid}...')
            X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
            y = pd.read_csv(output_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            train(X, y, model_path / f'sub_{sub}_vid_{vid}', multiprocessing.cpu_count(), args.num_gpus)
            logging.info(f'finish sub {sub} vid {vid}.')
