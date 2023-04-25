import pandas as pd
import multiprocessing
import numpy as np
from pathlib import Path
from io_generator import load_data_dict, load_model_dict
from utils import check_dir
from autogluon.tabular import TabularDataset, TabularPredictor
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
                        logging.FileHandler(f"../log/{log_filename}.log"),
                        logging.StreamHandler()
                        ],
                    level=logging.INFO
                    )

def train(X, y, target, drop, model_path, num_cpus, num_gpus):
    train_data = pd.concat([X, y.drop([drop], axis=1)], axis=1)
    train_data = TabularDataset(train_data)
    predictor = TabularPredictor(label=target, problem_type='regression', path=str(model_path) + '_' + target, verbosity=0).fit(train_data, ag_args_fit={'num_cpus': num_cpus, 'num_gpus': num_gpus})
    logging.info('arousal model fitted.')

prefix = '../'

data_dict = load_data_dict()
model_dict = load_model_dict(data_dict)

scenario = 3

input_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'train/physiology/arousal_10162022.csv'
output_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'train/annotations/arousal_10162022.csv'
model_path = Path(prefix) / f'models/scenario_{scenario}/arousal_10162022' 
check_dir(model_path)

logging.info(f'start arousal 10162022...')
X = pd.read_csv(input_path, index_col='time')
y = pd.read_csv(output_path, index_col='time')
train(X, y, target='arousal', drop='valence', model_path=model_path, num_cpus=multiprocessing.cpu_count(), num_gpus=args.num_gpus)
logging.info(f'finish arousal 10162022')
