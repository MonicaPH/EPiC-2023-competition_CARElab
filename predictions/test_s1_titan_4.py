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

def test(X, model_path, save_path):
    test_data = TabularDataset(X)

    predictor_arousal = TabularPredictor(str(model_path) + '_arousal')
    pred_arousal = predictor_arousal.predict(test_data)
    logging.info('arousal predicted')

    predictor_valence = TabularPredictor(str(model_path) + '_valence')
    pred_valence = predictor_valence.predict(test_data)
    logging.info('valence predicted')

    predictions = pd.DataFrame({'time': X.index, 'arousal': pred_arousal, 'valence': pred_valence}, index='time')
    predictions.to_csv(save_path)

prefix = '../'

scenario = 1
subs = [20, 22, 26, 28, 29]
vids = [1, 9, 10, 11, 13, 14, 18, 20]
num_gpus = 2

input_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'test' / 'physiology'
model_path = Path(prefix) / f'models/scenario_{scenario}'
save_path = Path(prefix) / f'predictions/scenario_{scenario}'
check_dir(model_path, save_path)

for sub in subs:
    for vid in vids:
        logging.info(f'start sub {sub} vid {vid}...')
        X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
        test(X, model_path / f'sub_{sub}_vid_{vid}', save_path / f'sub_{sub}_vid_{vid}')
        logging.info(f'finish sub {sub} vid {vid}.')
